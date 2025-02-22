from project.exception.exception import ProjectException
import sys,os
from project.logging.logger import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from imblearn.over_sampling import SMOTE  


from project.constants.training_pipeline import TARGET_COLUMN
from project.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from project.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact

from project.entity.config_entity import DataTransformationConfig
from project.utils.main_utils.utils import save_object,save_numpy_Array


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
             data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise ProjectException(e, sys)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ProjectException(e, sys)
    
    def get_data_transformer_object(cls)-> Pipeline:
        try:
            binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
            three_categorical_columns = ['PaymentMethod','Contract','StreamingMovies','StreamingTV','TechSupport','OnlineSecurity','OnlineBackup','DeviceProtection','MultipleLines','InternetService']
            numeric_columns = ['tenure','MonthlyCharges','TotalCharges']

            imputer = SimpleImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            

            num_pipeline = Pipeline(steps=[('imputer', imputer),('std_scaler', StandardScaler())])
            binary_pipeline = Pipeline(steps=[('encoder', OneHotEncoder(drop='first', sparse_output=False))])
            cat_pipeline = Pipeline(steps=[('encoder',OneHotEncoder(drop='first',sparse_output=False) )])
            preprocessor = ColumnTransformer(transformers=[('num', num_pipeline, numeric_columns),
                                                           ('binary', binary_pipeline, binary_columns),
                                                           ('cat', cat_pipeline, three_categorical_columns)])

            return preprocessor
        except Exception as e:
            raise ProjectException(e, sys)

    

    def initiate_data_transformation(self)-> DataTransformationArtifact:
        logging.info("Initiating data transformation...")
        try:
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            # Konwersja 'TotalCharges' na float
            train_df['TotalCharges'] = train_df['TotalCharges'].replace(' ', np.nan).astype(float)
            test_df['TotalCharges'] = test_df['TotalCharges'].replace(' ', np.nan).astype(float)

            # Podział na cechy i target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_train_df = train_df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
            
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_test_df = test_df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})

            # Przetwarzanie danych
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            
            input_feature_train_transformed = preprocessor_object.transform(input_feature_train_df)
            input_feature_test_transformed = preprocessor_object.transform(input_feature_test_df)

            # Zastosuj SMOTE tylko do danych treningowych
            sm = SMOTE(random_state=42)
            input_feature_train_resampled, target_train_resampled = sm.fit_resample(
                input_feature_train_transformed, 
                target_train_df
            )

            input_feature_test_resampled, target_test_resampled = sm.fit_resample(
                input_feature_test_transformed,
                target_test_df
            )

            # Przygotuj finalne tablice numpy
            train_arr = np.c_[input_feature_train_resampled, target_train_resampled]
            test_arr = np.c_[input_feature_test_resampled, target_test_resampled]

            logging.info("Data transformation completed")

            # Zapis wyników
            save_numpy_Array(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_Array(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise ProjectException(e, sys)

