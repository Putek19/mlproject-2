import os,sys
from project.logging.logger import logging
from project.exception.exception import ProjectException

from project.components.data_ingestion import DataIngestion
from project.components.data_validation import DataValidation
from project.components.data_transformation import DataTransformation
from project.components.model_trainer import ModelTrainer
from project.constants.training_pipeline import TRAINING_BUCKET_NAME

from project.entity.config_entity import (DataIngestionConfig,
DataValidationConfig,DataTransformationConfig,ModelTrainerConfig, TrainingPipelineConfig)

from project.entity.artifact_entity import (DataIngestionArtifact, 
                DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact)
from project.components.cloud.s3_syncer import S3Sync

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()
    
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            logging.info("Data ingestion started")
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact  = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise ProjectException(e, sys)
    
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.data_validation_config = DataValidationConfig(self.training_pipeline_config)
            logging.info("Data validation started")
            data_validation = DataValidation(data_validation_config=self.data_validation_config,data_ingestion_artifact= data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed and artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise ProjectException(e, sys)

    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            logging.info("Data transformation started")
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=self.data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed and artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ProjectException(e, sys)
    
    def start_model_training(self, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            logging.info("Model training started")
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact, model_trainer_config=self.model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_training()
            logging.info(f"Model training completed and artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ProjectException(e, sys)
        
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise ProjectException(e, sys)
    
    def sync_saved_model_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.model_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise ProjectException(e, sys)

    
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_training(data_transformation_artifact)
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_to_s3()
            return model_trainer_artifact
        except Exception as e:
            raise ProjectException(e, sys)