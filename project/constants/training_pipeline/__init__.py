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