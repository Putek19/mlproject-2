import os
import sys
import numpy as np
import pandas as pd

"""DEFINING COMMON CONSTANT VARIABLE FOR TRAINING PIPELINE"""
TARGET_COLUMN:str = "Churn"
PIPELINE_NAME:str = "ChurnPredict"
ARTIFACT_DIR:str = "Artifacts"
FILE_NAME:str = "churn_data.csv"


TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"

SAVED_MODEL_DIR =os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"





"""DATA INGESTION RELATED CONSTANTS"""
DATA_INGESTION_COLLECTION_NAME:str = "ChurnData"
DATA_INGESTION_DATABASE_NAME: str = "KUBAN"
DATA_INGESTION_DIR_NAME:str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str = "feature_store"
DATA_INGESTION_INGESTED_DIR:str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float = 0.2

SCHEMA_FILE_PATH = os.path.join("data_schema","schema.yaml")


""""DATA VALIDATION RELATED CONSTANTS"""
DATA_VALIDATION_DIR_NAME:str = "data_validation"
DATA_VALIDATION_VALID_DIR:str = "validated"
DATA_VALIDATION_INVALID_DIR:str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"



"""DATA TRANSFORMATION RELATED CONSTANTS"""
DATA_TRANSFORMATION_DIR_NAME:str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str = "transformed_object"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

DATA_TRANSFORMATION_IMPUTER_PARAMS = {
    "strategy": "mean"
}

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"
    