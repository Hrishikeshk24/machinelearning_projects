# Module 5 - Prescriptive Analytics

# In this script we prepare data to export to Excel and use in
# optimization models.

# The main extra parameter we need from the data is an estimate of the
# standard deviation of the return of each loan, which we can use
# to quantify the risk of the loan.

# To do this we use k-means clustering to group loans into groups of similar
# loans, then assign the return standard deviation in each group as the
# standard deviation of each loan.

#!! TO DO
#! (A) Change the number of clusters based on your assessment.
#! (B) Use the best model for predicting the returns of your chosen method
#       from the last module and replace the lasso model here with that.
#! (C) You can output any additional fields about the test loans you may want 
#        to examine later if they will help you explain your final loans picked
#        by your optimization model.


#===============================================================================
#Step 1:  computing risk of loans.

## Load saved data
# We will drop the first column that read.csv adds as it isn't necessary
loans.final <- read.csv('ret_loans_data.csv')[-1]

# final set of features to use for default probability and predicting returns
features <- c('home_ownership','term','purpose','emp_length',
              'loan_amnt','funded_amnt','annual_inc','dti','revol_bal','delinq_2yrs',
              'pub_rec','revol_util', 'cr_hist','int_rate','grade')
summary(loans.final)

## add the employment length, credit history, and default features
# Use readr package to parse numbers from strings easily
if(!require(readr)) { install.packages("readr") }
library(readr)
# convert employment length to numeric values
loans.final$emp_length <- as.character(loans.final$emp_length)
emp_length <- parse_number(loans.final[,'emp_length'])

loans.final['emp_length'] <- emp_length

# Add credit history and default columns
loans.final[,'cr_hist'] <- (loans.final$issue_d - loans.final$earliest_cr_line) /30

loans.final <- loans.final[which(loans.final$loan_status %in% c('Fully Paid','Charged Off','Default')),]
loans.final[,'default'] <- as.factor(as.integer(loans.final$loan_status %in% c('Charged Off','Default')))

# remove NA rows
loans.final <- na.omit(loans.final)


## set up the data to carry out a k-means clustering to find std dev's for loan returns
# columns to use in the clustering
#! You can change the set of columns used below as you see fit
cluster_columns <- c('int_rate', 'annual_inc','loan_amnt','emp_length','funded_amnt','dti','pub_rec','cr_hist')

# let's save the mean and std dev so we can reverse the scaling
loans.mean=colMeans(loans.final[,cluster_columns])
loans.sd=apply(loans.final[,cluster_columns],2,sd)

# scale the data set before applying k-means
loans.final[,cluster_columns] <- scale(loans.final[,cluster_columns])

## prepare train/test sets
set.seed(2314513)
N <- nrow(loans.final)
fraction <- 0.7                     # fraction of examples to put in training set
rand_values <- runif(N)
train_idxs <- rand_values <= fraction
test_idxs <- rand_values > fraction

# pick out the training set to make the k-means model
loans.train <- loans.final[train_idxs,]


# Choose a value of k for the k means clustering
# Now we explore clustering for different values of K
K_max <- 100
wcss <- numeric(K_max)
for(K in 1:K_max) {
  clustersK <- kmeans(loans.train[,cluster_columns],centers=K)
  wcss[K] <- clustersK$tot.withinss
}
plot(1:K_max,wcss,main="Within Cluster Sum of Squares vs K",xlab = 'K',ylab = 'WCSS')
lines(1:K_max,wcss)

# Trying k = 5 

#! (A) Change the number of clusters based on your assessment.
K <- 5
clusters <- kmeans(loans.train[,cluster_columns],centers=K,nstart=50)


#Tip: Since we are no longer interested in interpret the clusters (like in module 2)
# you might want to use a large value of k. If you do so, check that your do not 
# end up with very small clusters.

## Now predict cluster centers for the test set

# load a package for fast nearest neighbor look-ups
if(!require(FNN)) { install.packages("FNN") }
library(FNN)

# assign cluster centers
# this runs a nearest neighbor algorithm with the cluster centroids as input data
# and the loans as query points.  We extract only the index of the assigned centers.
loans.clusters <- get.knnx(clusters$centers,loans.final[,cluster_columns],1)$nn.index[,1]

## Use the clustering to assign a standard deviation to each loan

# names of return columns and corresponding columns for estimated std deviations
# important that these two have the same length for the "for" loop below
ret_cols <- c('ret_PESS','ret_OPT','ret_INTa','ret_INTb','ret_INTc')
std_dev_cols <- c('std_PESS','std_OPT','std_INTa','std_INTb','std_INTc')

# Initialize the values for these columns
for (col in std_dev_cols) {
  loans.final[,col] <- NA
}
# Calculate the standard deviation for each group and return type
st_dev_values <- as.data.frame(matrix(nrow=K,ncol=5))
colnames(st_dev_values) <- ret_cols

# Note that stdevn values are calculated only from the training data set
for (i in 1:K) {
  for (j in ret_cols) {
    st_dev_values[i,j] = sd(loans.train[clusters$cluster == i,j])
  }
}

# Fill the std dev values for each type for each data point
for (i in 1:nrow(loans.final)) {
  loans.final[i,"std_PESS"] <- st_dev_values[loans.clusters[i], "ret_PESS"]
  loans.final[i,"std_OPT"]  <- st_dev_values[loans.clusters[i], "ret_OPT"]
  loans.final[i,"std_INTa"] <- st_dev_values[loans.clusters[i], "ret_INTa"]
  loans.final[i,"std_INTb"] <- st_dev_values[loans.clusters[i], "ret_INTb"]
  loans.final[i,"std_INTc"] <- st_dev_values[loans.clusters[i], "ret_INTc"]
}

# Re-scale the continuous features
# translate the values back to the original scale
loans.final[,cluster_columns]=sweep(loans.final[,cluster_columns],MARGIN=2,loans.sd[cluster_columns],'*')       # step 1) scale: multiply the corresponding sd
loans.final[,cluster_columns]=sweep(loans.final[,cluster_columns],MARGIN=2,loans.mean[cluster_columns],'+')  # step 2) shift: add the original mean

# pick out the test set
loans.test <- loans.final[test_idxs,]
# we can reuse the indexes above to generate train/test sets for each return method


#===============================================================================
#Step 2:  Run the LASSO model for predicting returns on the test data set
#! (B) Use the best model for predicting the returns of your chosen method
#       from the last module and replace the lasso model here with that.

# Pessimistic return method
ret_col <- "ret_PESS"
## For other return methods, comment the above and uncomment the one you want to use
#######Optimistic return method ##############################
# ret_col <- "ret_OPT"
####### Interest return method a #############################
# ret_col <- "ret_INTa"
####### Interest return method b #############################
# ret_col <- "ret_INTb"
####### Interest return method c #############################
# ret_col <- "ret_INTc"

regression.train <- loans.final[train_idxs,c(ret_col,features)]
regression.test <- loans.final[test_idxs,c(ret_col,features)]

if (!require(glmnet)){ install.packages('glmnet') }; library("glmnet")
if (!require(glmnetUtils)){ install.packages("glmnetUtils")}; library("glmnetUtils")
## L2-Regularized linear regression, i.e. LASSO

# use alpha = 1 to select LASSO in glmnet
lasso.mod <- glmnet(ret_PESS ~ .-funded_amnt,data=regression.train,alpha=1)
summary(lasso.mod)
plot(lasso.mod)
# Generate a cv record for the various lambdas
lassoreg <- cv.glmnet(ret_PESS ~ .   -funded_amnt,data=regression.train,alpha=1)
summary(lassoreg)
plot(lassoreg)
coef(lassoreg,s="lambda.min")

# Compute Mean-Squared-Prediction-Error (MSPE) of best LASSO model in test set
(lassoreg_mspe <- mean((regression.test$ret_PESS - predict(lassoreg,regression.test,s="lambda.min")) ^ 2))

test.predn <- data.frame(rownames(regression.test))

test.predn$lassoret_PESS <- predict(lassoreg,regression.test,s="lambda.min")

#===============================================================================
#Step 3: Export data to excel

loans.test.aug <- cbind(loans.test,test.predn)

# Change the below to the actual return method you used
#  The alternate pairs would be 'lassoret_OPT' and 'std_OPT' etc. 
output_cols <- c('lassoret_PESS','std_PESS','funded_amnt','grade','int_rate','default','total_pymnt','recoveries')
#! (C) You can output any additional fields about the test loans you may want 
#        to examine later if they will help you explain your final loans picked
#        by your optimization model.

# selection of rows to output
sample_size <- 2000 # maybe no more than 10,000, or the size of the test set
set.seed(98237)
output_rows <- sample(nrow(loans.test),sample_size)

# create a single data file for all the returns
write.csv(loans.test.aug[output_rows,output_cols],'loans_pred_opti_data.csv')


#===============================================================================
# Step 4: Use the output file to solve the suggested optimization model
#
# Option 1: Use Excel
# You can use the output returns and stdev values for the test loans as the starting point
# The Excel models in M5-models.xlsx are a good place to get an initial set of models running
# You can change the constraints and parameters in this model to arrive at your final
#  portfolio recommended by optimization.

# Option 2: Use the lpSolve package in R to run the optimization model
if(!require(lpSolve)) { install.packages("lpSolve") }

loans.pred.opti.data <- loans.test.aug[output_rows,output_cols]

# (1) Simple Knapsack Model

# Set coefficients of the objective function
f.obj <- loans.pred.opti.data$funded_amnt*loans.pred.opti.data$lassoret_PESS[1:nrow(loans.pred.opti.data)]
# Set matrix corresponding to coefficients of constraints by rows
f.con <- matrix(c(loans.pred.opti.data$funded_amnt,
                  rep(1, times=nrow(loans.pred.opti.data)),
                  rep(1, times=nrow(loans.pred.opti.data))), 
                nrow = 3, byrow = TRUE)
# Set inequality/equality signs
f.dir <- c("<=",
           "<=",
           ">=")
# Set right hand side coefficients
f.rhs <- c(500000, # budget usage: max
           150, # total loan count: max
           135) # total loan count: min
# Optimization
skm <- lp("max", f.obj, f.con, f.dir, f.rhs, int.vec = 1:nrow(loans.pred.opti.data), all.bin = TRUE)
# total revenue
skm
# total number of loans picked
sum(skm$solution)
# The subset of loans picked by the model
loans.pred.opti.skm.solution <- loans.pred.opti.data[skm$solution==1,]

# (2) Multiple Knapsack Model

# Set coefficients of the objective function
f.obj <- loans.pred.opti.data$funded_amnt*loans.pred.opti.data$lassoret_PESS[1:nrow(loans.pred.opti.data)]
# Set matrix corresponding to coefficients of constraints by rows
f.con <- matrix(c(loans.pred.opti.data$funded_amnt,
                  rep(1, times=nrow(loans.pred.opti.data)),
                  rep(1, times=nrow(loans.pred.opti.data)),
                  loans.pred.opti.data$std_PESS), 
                nrow = 4, byrow = TRUE)
# Set inequality/equality signs
f.dir <- c("<=",
           "<=",
           ">=",
           "<=")
# Set right hand side coefficients
f.rhs <- c(500000, # budget usage: max
           150, # total loan count: max
           15, # total loan count: min
           10) # risk budget: max
# Optimization
mkm <- lp("max", f.obj, f.con, f.dir, f.rhs, int.vec = 1:nrow(loans.pred.opti.data), all.bin = TRUE)
# total revenue
mkm
# total loan picked
sum(mkm$solution)

# (3) Markowitz Model

# Set Markowitz coefficients of the objective function
sensitivity <- 0.1
f.obj <- (loans.pred.opti.data$lassoret_PESS-sensitivity*loans.pred.opti.data$std_PESS)*loans.pred.opti.data$funded_amnt
# Set matrix corresponding to coefficients of constraints by rows
f.con <- matrix(c(loans.pred.opti.data$funded_amnt, 
                  rep(1, times=nrow(loans.pred.opti.data)), 
                  rep(1, times=nrow(loans.pred.opti.data))), 
                nrow = 3, byrow = TRUE)
# Set inequality/equality signs
f.dir <- c("<=",
           "<=",
           ">=")
# Set right hand side coefficients
f.rhs <- c(400000, # budget usage: max
           150, # total loan count: max
           135) # total loan count: min
# Optimization
mm <- lp("max", f.obj, f.con, f.dir, f.rhs, int.vec = 1:nrow(loans.pred.opti.data), all.bin = TRUE)
# total revenue
mm
# total loan picked
sum(mm$solution)




