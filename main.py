from project.components.data_ingestion import DataIngestion
from project.components.data_validation import DataValidation
from project.components.data_transformation import DataTransformation
from project.exception.exception import ProjectException
from project.logging.logger import logging
from project.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from project.entity.config_entity import TrainingPipelineConfig
import sys
if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info("Data ingestion started")
        data_ingestion_artifact  = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")
        print(data_ingestion_artifact)
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
        logging.info("Data validation started")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)
        logging.info("Data transformation started")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, 
                                                 data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed")
        print(data_transformation_artifact)

    except Exception as e:
        raise ProjectException(e, sys)