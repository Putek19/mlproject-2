from project.components.data_ingestion import DataIngestion
from project.exception.exception import ProjectException
from project.logging.logger import logging
from project.entity.config_entity import DataIngestionConfig
from project.entity.config_entity import TrainingPipelineConfig
import sys
if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info("Data ingestion started")
        data_ingestion_artifact  = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
    except Exception as e:
        raise ProjectException(e, sys)