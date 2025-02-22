import sys
import os

import certifi

from project.components.cloud.s3_helper import S3Helper
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
print(mongo_db_url)

import pymongo
from project.exception.exception import ProjectException
from project.logging.logger import logging
from project.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from project.utils.main_utils.utils import load_object

from project.utils.ml_utils.model.estimator import ChurnModel

client = pymongo.MongoClient(mongo_db_url)

from project.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME,DATA_INGESTION_COLLECTION_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags = ["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")


    except Exception as e:
        raise ProjectException(e,sys)


from io import BytesIO
from botocore.exceptions import NoCredentialsError
from project.constants.training_pipeline import TRAINING_BUCKET_NAME
import boto3

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        s3_helper = S3Helper()  # Initialize helper
        bucket_name = TRAINING_BUCKET_NAME
        prefix = "final_model/"

        # Get latest model key
        latest_model_prefix = s3_helper.get_latest_model_key(bucket_name, prefix)
        
        # Load model and preprocessor
        preprocessor = s3_helper.load_from_s3(
            bucket_name,
            f"{latest_model_prefix}preprocessor.pkl"
        )
        model = s3_helper.load_from_s3(
            bucket_name,
            f"{latest_model_prefix}model.pkl"
        )

        # Rest of your prediction logic
        churn_model = ChurnModel(prepocessor=preprocessor, model=model)
        df = pd.read_csv(file.file)
        y_pred = churn_model.predict(df)

       

    
       
        df['predicted_column'] = y_pred
    
        
        return templates.TemplateResponse("table.html", {"request": request, "table": df.to_html(classes="table table-striped")})
    except Exception as e:
        raise ProjectException(e, sys)



if __name__=="__main__":
    try:
        app_run(app, host= "localhost",port=8000)
    except Exception as e:
        raise ProjectException(e,sys)