#!/usr/bin/env python3
"""
PD / LGD Credit Risk Dashboard
================================
Generates a self-contained static HTML dashboard backed by a SQLite database.
The dashboard is only rebuilt when underlying data has changed since the last run.

Usage
-----
  python dashboard.py              # Build if stale, skip if current
  python dashboard.py --open       # Build if stale, then open in browser
  python dashboard.py --force      # Force a full rebuild
  python dashboard.py --seed       # Reload synthetic sample data (clears existing)
  python dashboard.py --seed --open

Extending
---------
  1. Add new rows to any SQLite table via DataManager.insert_*()
  2. Add new KPIs in MetricsCalculator
  3. Add new Plotly figures in ChartBuilder
  4. Wire them into DashboardBuilder.build()
  The next run will detect the data change and rebuild automatically.
"""

import argparse
import hashlib
import os
import sqlite3
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "data" / "pd_lgd.db"
OUT_PATH   = BASE_DIR / "output" / "dashboard.html"
HASH_FILE  = BASE_DIR / "output" / ".data_hash"

# ── Design tokens ──────────────────────────────────────────────────────────────

C = dict(
    primary   = "#1f4e79",
    secondary = "#2e75b6",
    accent    = "#ed7d31",
    success   = "#70ad47",
    danger    = "#c00000",
    muted     = "#7f7f7f",
)

_LAYOUT = dict(
    paper_bgcolor = "white",
    plot_bgcolor  = "#f8f9fa",
    font          = dict(family="Segoe UI, Arial", size=12, color="#333"),
    margin        = dict(l=45, r=20, t=50, b=40),
    legend        = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

class DataManager:
    """All SQLite read/write operations. Tables are created on first use."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── connection ────────────────────────────────────────────────────────────

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ── schema ────────────────────────────────────────────────────────────────

    def _init_schema(self):
        ddl = """
        CREATE TABLE IF NOT EXISTS pd_observations (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            observation_date TEXT    NOT NULL,   -- ISO date, e.g. '2024-03-01'
            rating_grade     TEXT    NOT NULL,   -- AAA … CCC
            industry         TEXT    NOT NULL,
            n_obligors       INTEGER NOT NULL,
            n_defaults       INTEGER NOT NULL,
            pd_model         REAL    NOT NULL,   -- model-predicted PD
            created_at       TEXT    DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS lgd_observations (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            default_date         TEXT    NOT NULL,
            collateral_type      TEXT    NOT NULL,
            industry             TEXT    NOT NULL,
            lgd_realized         REAL    NOT NULL,   -- 0–1
            lgd_model            REAL    NOT NULL,   -- 0–1
            recovery_months      INTEGER,
            exposure_at_default  REAL    NOT NULL,
            created_at           TEXT    DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS model_performance (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name   TEXT    NOT NULL,
            eval_date    TEXT    NOT NULL,
            gini         REAL,
            auroc        REAL,
            ks_statistic REAL,
            brier_score  REAL,
            created_at   TEXT    DEFAULT CURRENT_TIMESTAMP
        );
        """
        with self._conn() as c:
            c.executescript(ddl)

    # ── reads ─────────────────────────────────────────────────────────────────

    def load_pd(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM pd_observations  ORDER BY observation_date", c)

    def load_lgd(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM lgd_observations ORDER BY default_date", c)

    def load_perf(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM model_performance ORDER BY eval_date", c)

    def has_data(self) -> bool:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM pd_observations").fetchone()[0] > 0

    # ── writes ────────────────────────────────────────────────────────────────

    def insert_pd(self, df: pd.DataFrame):
        """Append PD observations. df must match pd_observations columns (minus id/created_at)."""
        with self._conn() as c:
            df.to_sql("pd_observations", c, if_exists="append", index=False)

    def insert_lgd(self, df: pd.DataFrame):
        """Append LGD observations."""
        with self._conn() as c:
            df.to_sql("lgd_observations", c, if_exists="append", index=False)

    def insert_perf(self, df: pd.DataFrame):
        """Append model-performance records."""
        with self._conn() as c:
            df.to_sql("model_performance", c, if_exists="append", index=False)

    def clear_all(self):
        with self._conn() as c:
            c.executescript("DELETE FROM pd_observations; DELETE FROM lgd_observations; DELETE FROM model_performance;")

    # ── change detection ──────────────────────────────────────────────────────

    def data_hash(self) -> str:
        """MD5 over row counts + latest timestamps — fast proxy for data changes."""
        with self._conn() as c:
            row = c.execute("""
                SELECT
                  (SELECT COUNT(*)      FROM pd_observations),
                  (SELECT MAX(created_at) FROM pd_observations),
                  (SELECT COUNT(*)      FROM lgd_observations),
                  (SELECT MAX(created_at) FROM lgd_observations),
                  (SELECT COUNT(*)      FROM model_performance),
                  (SELECT MAX(created_at) FROM model_performance)
            """).fetchone()
        return hashlib.md5(str(row).encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA GENERATOR  (replace or skip in production)
# ══════════════════════════════════════════════════════════════════════════════

def generate_sample_data(dm: DataManager):
    """
    Populate the database with synthetic but realistic PD/LGD data covering
    12 quarters (2022–2024) across rating grades, industries, and collateral types.
    """
    rng = np.random.default_rng(42)

    GRADES    = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    INDUSTRIES = ["Financial", "Technology", "Healthcare", "Energy",
                  "Real Estate", "Retail", "Manufacturing"]
    COLLATERAL = ["Real Estate", "Equipment", "Receivables",
                  "Inventory", "Unsecured", "Securities"]

    BASE_PD  = dict(AAA=0.001, AA=0.003, A=0.008, BBB=0.02, BB=0.05, B=0.12, CCC=0.25)
    BASE_LGD = {"Real Estate": 0.25, "Equipment": 0.35, "Receivables": 0.40,
                "Inventory": 0.50, "Unsecured": 0.65, "Securities": 0.20}

    # ── PD observations (quarterly, all grade × industry combos) ──────────────
    pd_rows = []
    for year in range(2022, 2025):
        for q in range(1, 5):
            obs_date = f"{year}-{q*3:02d}-01"
            cycle    = 1.0 + 0.35 * np.sin((year - 2022 + q / 4) * np.pi)
            for grade in GRADES:
                for ind in INDUSTRIES:
                    n_obligors = int(rng.integers(50, 500))
                    pd_act     = float(np.clip(BASE_PD[grade] * cycle * rng.uniform(0.7, 1.3), 0, 1))
                    n_defaults = int(rng.binomial(n_obligors, pd_act))
                    pd_model   = float(np.clip(BASE_PD[grade] * rng.uniform(0.88, 1.12), 0, 1))
                    pd_rows.append(dict(
                        observation_date=obs_date, rating_grade=grade, industry=ind,
                        n_obligors=n_obligors, n_defaults=n_defaults, pd_model=pd_model
                    ))
    dm.insert_pd(pd.DataFrame(pd_rows))

    # ── LGD observations (1 000 individual defaults) ──────────────────────────
    lgd_rows = []
    for _ in range(1_000):
        yr   = rng.integers(2022, 2025)
        mo   = rng.integers(1, 13)
        coll = rng.choice(COLLATERAL)
        ind  = rng.choice(INDUSTRIES)
        lgd_r = float(np.clip(rng.beta(2, 4) + BASE_LGD[coll] * 0.4, 0.05, 0.98))
        lgd_m = float(np.clip(BASE_LGD[coll] * rng.uniform(0.85, 1.15), 0.05, 0.95))
        lgd_rows.append(dict(
            default_date=f"{yr}-{mo:02d}-01",
            collateral_type=coll, industry=ind,
            lgd_realized=lgd_r, lgd_model=lgd_m,
            recovery_months=int(rng.integers(3, 49)),
            exposure_at_default=float(rng.uniform(10_000, 5_000_000))
        ))
    dm.insert_lgd(pd.DataFrame(lgd_rows))

    # ── Model performance (quarterly) ─────────────────────────────────────────
    perf_rows = []
    for year in range(2022, 2025):
        for q in range(1, 5):
            perf_rows.append(dict(
                model_name="PD_Model_v2",
                eval_date=f"{year}-{q*3:02d}-01",
                gini         = float(rng.uniform(0.55, 0.72)),
                auroc        = float(rng.uniform(0.77, 0.86)),
                ks_statistic = float(rng.uniform(0.38, 0.52)),
                brier_score  = float(rng.uniform(0.04, 0.08)),
            ))
    dm.insert_perf(pd.DataFrame(perf_rows))
    print("  Sample data seeded into database.")


# ══════════════════════════════════════════════════════════════════════════════
# METRICS CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

class MetricsCalculator:
    """Derives all KPIs and aggregates from raw DataFrames."""

    GRADE_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

    def __init__(self, pd_df: pd.DataFrame, lgd_df: pd.DataFrame, perf_df: pd.DataFrame):
        self.pd_df   = pd_df.copy()
        self.lgd_df  = lgd_df.copy()
        self.perf_df = perf_df.copy()
        self._preprocess()

    def _preprocess(self):
        self.pd_df["observation_date"] = pd.to_datetime(self.pd_df["observation_date"])
        self.pd_df["pd_actual"]        = self.pd_df["n_defaults"] / self.pd_df["n_obligors"]
        self.lgd_df["default_date"]    = pd.to_datetime(self.lgd_df["default_date"])
        if not self.perf_df.empty:
            self.perf_df["eval_date"]  = pd.to_datetime(self.perf_df["eval_date"])

    # ── PD aggregates ─────────────────────────────────────────────────────────

    def pd_summary_kpis(self) -> dict:
        latest = self.pd_df[self.pd_df["observation_date"] == self.pd_df["observation_date"].max()]
        overall_pd = latest["n_defaults"].sum() / latest["n_obligors"].sum()
        tot_obl    = self.pd_df["n_obligors"].sum()
        tot_def    = self.pd_df["n_defaults"].sum()
        return {
            "Portfolio DR (Latest Period)": f"{overall_pd:.2%}",
            "Total Obligors (All Periods)": f"{tot_obl:,}",
            "Total Defaults (All Periods)": f"{tot_def:,}",
            "Overall Default Rate":         f"{tot_def/tot_obl:.2%}",
        }

    def pd_by_grade(self) -> pd.DataFrame:
        g = (self.pd_df
             .groupby("rating_grade")
             .agg(total_obligors=("n_obligors","sum"),
                  total_defaults=("n_defaults","sum"),
                  avg_model_pd  =("pd_model","mean"))
             .reset_index())
        g["actual_pd"]     = g["total_defaults"] / g["total_obligors"]
        g["pd_delta"]      = g["actual_pd"] - g["avg_model_pd"]
        g["rating_grade"]  = pd.Categorical(g["rating_grade"], self.GRADE_ORDER, ordered=True)
        return g.sort_values("rating_grade")

    def pd_trend(self) -> pd.DataFrame:
        g = (self.pd_df
             .groupby("observation_date")
             .agg(n_obligors=("n_obligors","sum"),
                  n_defaults=("n_defaults","sum"),
                  avg_model_pd=("pd_model","mean"))
             .reset_index())
        g["actual_pd"] = g["n_defaults"] / g["n_obligors"]
        return g.sort_values("observation_date")

    def pd_by_industry(self) -> pd.DataFrame:
        g = (self.pd_df
             .groupby("industry")
             .agg(total_obligors=("n_obligors","sum"),
                  total_defaults=("n_defaults","sum"))
             .reset_index())
        g["actual_pd"] = g["total_defaults"] / g["total_obligors"]
        return g.sort_values("actual_pd", ascending=False)

    def pd_heatmap(self) -> pd.DataFrame:
        """PD actual by grade × industry (for heatmap)."""
        g = (self.pd_df
             .groupby(["rating_grade","industry"])
             .apply(lambda df: df["n_defaults"].sum() / df["n_obligors"].sum(), include_groups=False)
             .reset_index(name="actual_pd"))
        g["rating_grade"] = pd.Categorical(g["rating_grade"], self.GRADE_ORDER, ordered=True)
        return g.sort_values("rating_grade")

    # ── LGD aggregates ────────────────────────────────────────────────────────

    def lgd_summary_kpis(self) -> dict:
        d = self.lgd_df
        avg_lgd      = d["lgd_realized"].mean()
        w_lgd        = np.average(d["lgd_realized"], weights=d["exposure_at_default"])
        recovery     = 1 - avg_lgd
        total_ead    = d["exposure_at_default"].sum()
        el           = w_lgd * total_ead   # simplified EL proxy
        return {
            "Avg LGD (Realized)":  f"{avg_lgd:.2%}",
            "EAD-Weighted LGD":    f"{w_lgd:.2%}",
            "Avg Recovery Rate":   f"{recovery:.2%}",
            "Total EAD":           f"${total_ead/1e6:.1f}M",
        }

    def lgd_by_collateral(self) -> pd.DataFrame:
        return (self.lgd_df
                .groupby("collateral_type")
                .agg(count             =("lgd_realized","count"),
                     avg_lgd_realized  =("lgd_realized","mean"),
                     avg_lgd_model     =("lgd_model","mean"),
                     median_lgd        =("lgd_realized","median"),
                     total_ead         =("exposure_at_default","sum"))
                .reset_index()
                .sort_values("avg_lgd_realized", ascending=False))

    def lgd_by_industry(self) -> pd.DataFrame:
        return (self.lgd_df
                .groupby("industry")
                .agg(count           =("lgd_realized","count"),
                     avg_lgd_realized=("lgd_realized","mean"),
                     avg_lgd_model   =("lgd_model","mean"))
                .reset_index()
                .sort_values("avg_lgd_realized", ascending=False))

    def lgd_trend(self) -> pd.DataFrame:
        df = self.lgd_df.copy()
        df["period"] = df["default_date"].dt.to_period("Q").astype(str)
        return (df.groupby("period")
                  .agg(avg_lgd_realized=("lgd_realized","mean"),
                       avg_lgd_model   =("lgd_model","mean"),
                       count           =("lgd_realized","count"))
                  .reset_index()
                  .sort_values("period"))

    def lgd_distribution(self) -> pd.Series:
        return self.lgd_df["lgd_realized"]

    def recovery_scatter_data(self) -> pd.DataFrame:
        return self.lgd_df[["collateral_type","recovery_months","lgd_realized"]].dropna()

    # ── Model performance ─────────────────────────────────────────────────────

    def model_perf_trend(self) -> pd.DataFrame:
        return self.perf_df.sort_values("eval_date") if not self.perf_df.empty else self.perf_df

    def model_perf_latest(self) -> dict:
        if self.perf_df.empty:
            return {}
        row = self.perf_df.sort_values("eval_date").iloc[-1]
        return {
            "Gini":         f"{row.get('gini', float('nan')):.3f}",
            "AUROC":        f"{row.get('auroc', float('nan')):.3f}",
            "KS Statistic": f"{row.get('ks_statistic', float('nan')):.3f}",
            "Brier Score":  f"{row.get('brier_score', float('nan')):.4f}",
        }


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class ChartBuilder:
    """Builds Plotly figures. Each method is independent and returns go.Figure."""

    # ── PD charts ─────────────────────────────────────────────────────────────

    @staticmethod
    def pd_grade_bar(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure([
            go.Bar(x=df["rating_grade"], y=df["actual_pd"],
                   name="Actual PD",   marker_color=C["danger"],
                   text=[f"{v:.2%}" for v in df["actual_pd"]], textposition="outside"),
            go.Bar(x=df["rating_grade"], y=df["avg_model_pd"],
                   name="Model PD",    marker_color=C["secondary"],
                   text=[f"{v:.2%}" for v in df["avg_model_pd"]], textposition="outside"),
        ])
        fig.update_layout(title="Actual vs Model PD by Rating Grade",
                          barmode="group", yaxis_tickformat=".1%", **_LAYOUT)
        return fig

    @staticmethod
    def pd_trend_line(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure([
            go.Scatter(x=df["observation_date"], y=df["actual_pd"],
                       name="Actual PD", mode="lines+markers",
                       line=dict(color=C["danger"], width=2)),
            go.Scatter(x=df["observation_date"], y=df["avg_model_pd"],
                       name="Model PD",  mode="lines+markers",
                       line=dict(color=C["secondary"], width=2, dash="dash")),
        ])
        fig.update_layout(title="Portfolio Default Rate — Trend",
                          yaxis_tickformat=".2%", **_LAYOUT)
        return fig

    @staticmethod
    def pd_industry_bar(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure(go.Bar(
            x=df["actual_pd"], y=df["industry"], orientation="h",
            marker_color=C["primary"],
            text=[f"{v:.2%}" for v in df["actual_pd"]], textposition="outside",
        ))
        fig.update_layout(title="Default Rate by Industry",
                          xaxis_tickformat=".1%", **_LAYOUT)
        return fig

    @staticmethod
    def pd_heatmap(df: pd.DataFrame) -> go.Figure:
        pivot = df.pivot_table(index="rating_grade", columns="industry",
                               values="actual_pd", aggfunc="mean")
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r",
            text=[[f"{v:.2%}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            colorbar=dict(tickformat=".0%"),
        ))
        fig.update_layout(title="PD Heatmap: Rating Grade × Industry", **_LAYOUT)
        return fig

    # ── LGD charts ────────────────────────────────────────────────────────────

    @staticmethod
    def lgd_collateral_bar(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure([
            go.Bar(x=df["collateral_type"], y=df["avg_lgd_realized"],
                   name="Realized LGD", marker_color=C["accent"],
                   text=[f"{v:.1%}" for v in df["avg_lgd_realized"]], textposition="outside"),
            go.Bar(x=df["collateral_type"], y=df["avg_lgd_model"],
                   name="Model LGD",    marker_color=C["secondary"],
                   text=[f"{v:.1%}" for v in df["avg_lgd_model"]], textposition="outside"),
        ])
        fig.update_layout(title="LGD: Realized vs Model by Collateral",
                          barmode="group", yaxis_tickformat=".0%", **_LAYOUT)
        return fig

    @staticmethod
    def lgd_distribution(series: pd.Series) -> go.Figure:
        fig = go.Figure(go.Histogram(
            x=series, nbinsx=30, marker_color=C["accent"], opacity=0.85,
        ))
        fig.add_vline(x=series.mean(), line_dash="dash", line_color=C["primary"],
                      annotation_text=f"Mean {series.mean():.1%}",
                      annotation_position="top right")
        fig.add_vline(x=series.median(), line_dash="dot", line_color=C["muted"],
                      annotation_text=f"Median {series.median():.1%}",
                      annotation_position="top left")
        fig.update_layout(title="LGD Distribution (Realized)",
                          xaxis=dict(tickformat=".0%", title="LGD"),
                          yaxis_title="Count", **_LAYOUT)
        return fig

    @staticmethod
    def lgd_trend_line(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure([
            go.Scatter(x=df["period"], y=df["avg_lgd_realized"],
                       name="Realized LGD", mode="lines+markers",
                       line=dict(color=C["accent"], width=2)),
            go.Scatter(x=df["period"], y=df["avg_lgd_model"],
                       name="Model LGD",    mode="lines+markers",
                       line=dict(color=C["secondary"], width=2, dash="dash")),
        ])
        fig.update_layout(title="LGD Trend (Quarterly)",
                          yaxis_tickformat=".0%", **_LAYOUT)
        return fig

    @staticmethod
    def lgd_industry_bar(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure([
            go.Bar(x=df["industry"], y=df["avg_lgd_realized"],
                   name="Realized LGD", marker_color=C["accent"],
                   text=[f"{v:.1%}" for v in df["avg_lgd_realized"]], textposition="outside"),
            go.Bar(x=df["industry"], y=df["avg_lgd_model"],
                   name="Model LGD",    marker_color=C["secondary"],
                   text=[f"{v:.1%}" for v in df["avg_lgd_model"]], textposition="outside"),
        ])
        fig.update_layout(title="LGD by Industry",
                          barmode="group", yaxis_tickformat=".0%", **_LAYOUT)
        return fig

    @staticmethod
    def recovery_scatter(df: pd.DataFrame) -> go.Figure:
        fig = px.scatter(
            df, x="recovery_months", y="lgd_realized", color="collateral_type",
            opacity=0.55,
            title="Recovery Timeline vs LGD by Collateral",
            labels={"recovery_months": "Months to Recovery", "lgd_realized": "LGD Realized"},
        )
        fig.update_layout(yaxis_tickformat=".0%", **_LAYOUT)
        return fig

    # ── Model performance charts ───────────────────────────────────────────────

    @staticmethod
    def model_perf_line(df: pd.DataFrame) -> go.Figure:
        if df.empty:
            return go.Figure()
        metrics = [
            ("gini",         "Gini",         C["primary"]),
            ("auroc",        "AUROC",         C["success"]),
            ("ks_statistic", "KS Statistic",  C["accent"]),
        ]
        fig = go.Figure()
        for col, label, color in metrics:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["eval_date"], y=df[col],
                    name=label, mode="lines+markers",
                    line=dict(color=color, width=2),
                ))
        fig.update_layout(title="Model Discrimination Metrics Over Time",
                          yaxis=dict(range=[0, 1]), **_LAYOUT)
        return fig

    @staticmethod
    def brier_score_bar(df: pd.DataFrame) -> go.Figure:
        if df.empty or "brier_score" not in df.columns:
            return go.Figure()
        fig = go.Figure(go.Bar(
            x=df["eval_date"].astype(str), y=df["brier_score"],
            marker_color=C["secondary"],
            text=[f"{v:.4f}" for v in df["brier_score"]], textposition="outside",
        ))
        fig.update_layout(title="Brier Score (lower = better)", **_LAYOUT)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# HTML ASSEMBLER
# ══════════════════════════════════════════════════════════════════════════════

class DashboardBuilder:
    """Combines metrics + charts into a single self-contained HTML file."""

    def __init__(self, metrics: MetricsCalculator):
        self.m  = metrics
        self.cb = ChartBuilder()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fig(fig: go.Figure) -> str:
        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _kpi(label: str, value: str, border_color: str = C["secondary"]) -> str:
        return f"""
        <div class="kpi-card" style="border-left-color:{border_color}">
          <div class="kpi-value">{value}</div>
          <div class="kpi-label">{label}</div>
        </div>"""

    @staticmethod
    def _table(df: pd.DataFrame, fmt: dict = None) -> str:
        d = df.copy()
        if fmt:
            for col, fn in fmt.items():
                if col in d.columns:
                    d[col] = d[col].map(fn)
        return d.to_html(index=False, classes="data-table", border=0)

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self, out_path: Path):
        m  = self.m
        fi = self._fig

        # Compute aggregates
        pd_grade    = m.pd_by_grade()
        pd_trend    = m.pd_trend()
        pd_ind      = m.pd_by_industry()
        pd_heat     = m.pd_heatmap()
        lgd_coll    = m.lgd_by_collateral()
        lgd_ind     = m.lgd_by_industry()
        lgd_trend   = m.lgd_trend()
        lgd_dist    = m.lgd_distribution()
        rec_data    = m.recovery_scatter_data()
        perf        = m.model_perf_trend()

        # KPI bars
        pd_kpi_html   = "".join(self._kpi(k, v, C["danger"])   for k, v in m.pd_summary_kpis().items())
        lgd_kpi_html  = "".join(self._kpi(k, v, C["accent"])   for k, v in m.lgd_summary_kpis().items())
        perf_kpi_html = "".join(self._kpi(k, v, C["success"])  for k, v in m.model_perf_latest().items())

        # Tables
        pct = lambda v: f"{v:.2%}"
        t_pd_grade = self._table(
            pd_grade[["rating_grade","total_obligors","total_defaults","actual_pd","avg_model_pd","pd_delta"]],
            fmt={"actual_pd": pct, "avg_model_pd": pct, "pd_delta": pct}
        )
        t_lgd_coll = self._table(
            lgd_coll[["collateral_type","count","avg_lgd_realized","avg_lgd_model","median_lgd"]],
            fmt={"avg_lgd_realized": pct, "avg_lgd_model": pct, "median_lgd": pct}
        )
        t_lgd_ind = self._table(
            lgd_ind[["industry","count","avg_lgd_realized","avg_lgd_model"]],
            fmt={"avg_lgd_realized": pct, "avg_lgd_model": pct}
        )

        gen_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        # ── HTML ──────────────────────────────────────────────────────────────
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PD / LGD Credit Risk Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Segoe UI",Arial,sans-serif;background:#f0f2f5;color:#333;min-height:100vh}}
/* ── Header ── */
header{{background:{C["primary"]};color:#fff;padding:16px 32px;display:flex;align-items:center;justify-content:space-between}}
header h1{{font-size:1.4rem;font-weight:600;letter-spacing:.02em}}
header span{{font-size:.8rem;opacity:.7}}
/* ── Nav tabs ── */
nav{{display:flex;gap:0;background:#fff;border-bottom:2px solid #dde3ec;padding:0 32px}}
nav button{{border:none;background:none;padding:13px 20px;cursor:pointer;font-size:.88rem;
            color:#666;border-bottom:3px solid transparent;transition:all .18s;white-space:nowrap}}
nav button.active,nav button:hover{{color:{C["secondary"]};border-bottom-color:{C["secondary"]};font-weight:600}}
/* ── Tab panels ── */
.tab{{display:none;padding:24px 32px 40px}}
.tab.active{{display:block}}
/* ── KPI row ── */
.kpi-row{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:22px}}
.kpi-card{{background:#fff;border-radius:8px;padding:16px 22px;flex:1;min-width:170px;
           box-shadow:0 1px 4px rgba(0,0,0,.07);border-left:4px solid {C["secondary"]}}}
.kpi-value{{font-size:1.55rem;font-weight:700;color:{C["primary"]}}}
.kpi-label{{font-size:.72rem;color:#888;margin-top:5px;text-transform:uppercase;letter-spacing:.05em}}
/* ── Grid ── */
.grid{{display:grid;gap:18px;margin-bottom:20px}}
.g2{{grid-template-columns:1fr 1fr}}
.g3{{grid-template-columns:1fr 1fr 1fr}}
.span2{{grid-column:span 2}}
.span3{{grid-column:1/-1}}
/* ── Cards ── */
.card{{background:#fff;border-radius:8px;padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,.07)}}
.card h3{{font-size:.82rem;color:#666;margin-bottom:12px;text-transform:uppercase;letter-spacing:.06em}}
/* ── Table ── */
.data-table{{width:100%;border-collapse:collapse;font-size:.84rem}}
.data-table th{{background:{C["primary"]};color:#fff;padding:9px 13px;text-align:left;font-weight:500}}
.data-table td{{padding:8px 13px;border-bottom:1px solid #eee}}
.data-table tr:hover td{{background:#edf3fc}}
/* ── Footer ── */
footer{{text-align:center;color:#bbb;font-size:.75rem;padding:20px}}
@media(max-width:860px){{.g2,.g3{{grid-template-columns:1fr}}.span2,.span3{{grid-column:span 1}}}}
</style>
</head>
<body>

<header>
  <h1>&#128202; PD / LGD Credit Risk Dashboard</h1>
  <span>Generated: {gen_ts}</span>
</header>

<nav>
  <button class="active" onclick="showTab('pd',this)">Probability of Default</button>
  <button onclick="showTab('lgd',this)">Loss Given Default</button>
  <button onclick="showTab('model',this)">Model Performance</button>
</nav>

<!-- ═══════════════════ PD TAB ═══════════════════ -->
<div id="tab-pd" class="tab active">
  <div class="kpi-row">{pd_kpi_html}</div>
  <div class="grid g2">
    <div class="card">{fi(ChartBuilder.pd_grade_bar(pd_grade))}</div>
    <div class="card">{fi(ChartBuilder.pd_trend_line(pd_trend))}</div>
    <div class="card">{fi(ChartBuilder.pd_industry_bar(pd_ind))}</div>
    <div class="card">{fi(ChartBuilder.pd_heatmap(pd_heat))}</div>
    <div class="card span2">
      <h3>PD by Rating Grade — Detail Table</h3>
      {t_pd_grade}
    </div>
  </div>
</div>

<!-- ═══════════════════ LGD TAB ═══════════════════ -->
<div id="tab-lgd" class="tab">
  <div class="kpi-row">{lgd_kpi_html}</div>
  <div class="grid g2">
    <div class="card">{fi(ChartBuilder.lgd_collateral_bar(lgd_coll))}</div>
    <div class="card">{fi(ChartBuilder.lgd_distribution(lgd_dist))}</div>
    <div class="card">{fi(ChartBuilder.lgd_trend_line(lgd_trend))}</div>
    <div class="card">{fi(ChartBuilder.lgd_industry_bar(lgd_ind))}</div>
    <div class="card span2">{fi(ChartBuilder.recovery_scatter(rec_data))}</div>
    <div class="card">
      <h3>LGD by Collateral — Detail Table</h3>
      {t_lgd_coll}
    </div>
    <div class="card">
      <h3>LGD by Industry — Detail Table</h3>
      {t_lgd_ind}
    </div>
  </div>
</div>

<!-- ═══════════════════ MODEL PERFORMANCE TAB ═══════════════════ -->
<div id="tab-model" class="tab">
  <div class="kpi-row">{perf_kpi_html}</div>
  <div class="grid g2">
    <div class="card span2">{fi(ChartBuilder.model_perf_line(perf))}</div>
    <div class="card span2">{fi(ChartBuilder.brier_score_bar(perf))}</div>
  </div>
</div>

<footer>PD / LGD Dashboard &mdash; {gen_ts} &mdash; Data: SQLite @ {DB_PATH}</footer>

<script>
function showTab(name, btn) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body>
</html>"""

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD CONTROLLER — skip rebuild if data unchanged
# ══════════════════════════════════════════════════════════════════════════════

def is_stale(dm: DataManager) -> bool:
    """Return True when a rebuild is needed (no output file, or data changed)."""
    if not OUT_PATH.exists():
        return True
    if not HASH_FILE.exists():
        return True
    return HASH_FILE.read_text().strip() != dm.data_hash()


def record_hash(dm: DataManager):
    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    HASH_FILE.write_text(dm.data_hash())


def run_build(dm: DataManager):
    print("  Loading data from database …")
    pd_df   = dm.load_pd()
    lgd_df  = dm.load_lgd()
    perf_df = dm.load_perf()

    print("  Computing metrics …")
    metrics = MetricsCalculator(pd_df, lgd_df, perf_df)

    print("  Assembling HTML …")
    DashboardBuilder(metrics).build(OUT_PATH)
    record_hash(dm)
    print(f"  Dashboard saved  →  {OUT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PD/LGD Credit Risk Dashboard generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python dashboard.py                # build if stale
  python dashboard.py --open         # build if stale, open in browser
  python dashboard.py --force        # always rebuild
  python dashboard.py --seed --open  # reload sample data, rebuild, open
        """
    )
    parser.add_argument("--force", action="store_true", help="Force a full rebuild")
    parser.add_argument("--open",  action="store_true", help="Open dashboard in browser after build")
    parser.add_argument("--seed",  action="store_true", help="(Re-)load synthetic sample data")
    args = parser.parse_args()

    print("\nPD / LGD Dashboard")
    print("=" * 44)

    dm = DataManager(DB_PATH)

    if args.seed:
        print("Seeding sample data (clearing existing) …")
        dm.clear_all()
        generate_sample_data(dm)
    elif not dm.has_data():
        print("No data found — seeding sample data …")
        generate_sample_data(dm)

    if args.force or is_stale(dm):
        print("Building dashboard …")
        run_build(dm)
    else:
        print(f"Dashboard is current — no rebuild needed.")
        print(f"  File: {OUT_PATH}")

    if args.open:
        webbrowser.open(OUT_PATH.as_uri())
        print("  Opened in browser.")
    else:
        print(f"\nOpen manually:\n  {OUT_PATH}")

    print()


if __name__ == "__main__":
    main()
