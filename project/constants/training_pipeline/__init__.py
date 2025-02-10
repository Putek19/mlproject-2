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