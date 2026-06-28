# 1. Library imports
import uvicorn
from fastapi import FastAPI
from CCFraudDet import CCFraud
import numpy as np
import pickle
import pandas as pd


# 2. Create the app object
app = FastAPI()
pickle_in = open("classifier_nov11.pkl","rb")
classifier=pickle.load(pickle_in)


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data.

@app.post('/predict')
def predict_fraud(data:CCFraud):
    data = data.dict()
    variable14=data['variable14']
    variable12=data['variable12']
    variable17=data['variable17']
    Amount=data['Amount']
   
    prediction = classifier.predict([[406,-2.312226542,1.951992011,-1.609850732,3.997905588,-0.522187865,-1.426545319,
                                      -2.537387306, 1.391657248,-2.770089277,-2.772272145,3.202033207, variable12,
                                      -0.595221881, variable14, 0.38972412, -1.14074718, variable17, -0.016822468,
                                      0.416955705, 0.126910559, 0.517232371, -0.035049369,-0.465211076,0.320198199,
                                      0.044519167, 0.177839798, 0.261145003, -0.143275875, Amount]])
    
    if(prediction[0]>0):
        prediction="Fraud Transaction"
    else:
        prediction="Not a Fraud Transaction"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload