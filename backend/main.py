import sys  
import json
import mlflow
import sklearn
import uvicorn
import numpy as np
import pandas as pd
from operator import index
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from preprocessing import date_processing, scale_data
from example_json.drought_info import DroughtModel

#API
app = FastAPI(title = "Model Tracking")
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Welcome": "to Drought Prediction Using MLOps app version 1."}


# Predict 
@app.post("/predict")
def predict():
    return {"predictions": 'yes'}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)


