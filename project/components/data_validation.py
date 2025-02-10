from project.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from project.entity.config_entity import DataValidationConfig
from project.exception.exception import ProjectException
from project.logging.logger import logging
from project.constants.training_pipeline import SCHEMA_FILE_PATH
from project.utils.main_utils.utils import read_yaml_file,write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import os,sys




class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact ,data_validation_config:DataValidationConfig):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise ProjectException(e, sys)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ProjectException(e, sys)
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Actual number of columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            else:
                return False
        except Exception as e:
            raise ProjectException(e, sys)
    def validate_number_of_numerical_columns(self, dataframe:pd.DataFrame)->bool:
        try:
            numerical_columns_true = len(self._schema_config["numerical_columns"])
            numerical_columns = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64']]
            logging.info(f"Required number of numerical columns: {numerical_columns_true}")
            logging.info(f"Actual number of numerical columns: {len(numerical_columns)}")
            if numerical_columns_true == len(numerical_columns):
                return True
            else:
                return False
        except Exception as e:
            raise ProjectException(e, sys)
    
    def detect_data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame,threshold = 0.05)->bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist= ks_2samp(d1, d2)
                if threshold < is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column:{
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(drift_report_file_path, report)
        except Exception as e:
            raise ProjectException(e, sys)

    def initiate_data_validation(self)-> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            #validate number of columns
            status = self.validate_number_of_columns(train_df)
            if not status:
                error_message = "Number of columns in training data is not as expected"
            status = self.validate_number_of_columns(test_df)
            if not status:
                error_message = "Number of columns in testing data is not as expected"
            #validate number of numerical columns
            status = self.validate_number_of_numerical_columns(train_df)
            if not status:
                error_message = "Number of numerical columns in training data is not as expected"
            status = self.validate_number_of_numerical_columns(test_df)
            if not status:
                error_message = "Number of numerical columns in testing data is not as expected"

            #validate drift
            status = self.detect_data_drift(train_df, test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
        except Exception as e:
            raise ProjectException(e, sys)
