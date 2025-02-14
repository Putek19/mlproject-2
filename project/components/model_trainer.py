import os,sys
from urllib.parse import urlparse
import mlflow
import numpy as np

from project.exception.exception import ProjectException
from project.logging.logger import logging

from project.entity.config_entity import ModelTrainerConfig
from project.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact

from project.utils.main_utils.utils import save_object,load_object,load_numpy_array_data,evaluate_models

from project.utils.ml_utils.model.estimator import ChurnModel
from project.utils.ml_utils.metric.classification_metric import get_classification_metrics

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold

import dagshub
dagshub.init(repo_owner='Putek19', repo_name='mlproject-2', mlflow=True)



class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact): 
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ProjectException(e, sys)
        
    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score

            

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")
        

    
    def train_model(self, X_train, y_train,X_test,y_test):
        try:
            models = {
                'RandomForest': RandomForestClassifier(verbose=1),
                'AdaBoost': AdaBoostClassifier(),
                'GradientBoosting': GradientBoostingClassifier(verbose=1),
                'LogisticRegression': LogisticRegression(verbose=1),
                'SVM': SVC(verbose=1),
                'XGBoost': XGBClassifier(verbose=1),
                'DecisionTree': DecisionTreeClassifier()
            }
            params = {
                'RandomForest': {'n_estimators': [50,100,150 ,200],'max_depth': [3,5,9,15,20],'criterion': ['gini', 'entropy','log_loss'],'max_features': ['auto', 'sqrt', 'log2']},
                'AdaBoost': {'n_estimators': [50,100,150,200], 'learning_rate': [0.001,0.01,0.1]},
                'GradientBoosting': {'n_estimators': [50,100,150 ,200],'max_depth': [3,5,9,15,20],'learning_rate': [0.001,0.01,0.1],'loss':['log_loss', 'exponential'],},
                'LogisticRegression': {'penalty': [ 'l2'], 'C': [0.01,0.1,1,0.0001,10], 'solver': ['liblinear','newton-cg', 'newton-cholesky', 'sag', 'saga']},
                'SVM': {'C': [0.01,0.1,1,0.0001,10], 'gamma': [0.01, 0.1, 1],'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                'XGBoost': {'n_estimators': [50,100,150 ,200],'max_depth': [3,5,9,15,20],'reg_lambda': [1,2,0.1,1.1],'sampling_method': ['uniform', 'gradient_based']},
                'DecisionTree': {'criterion': ['gini', 'entropy','log_loss'], 'max_depth': [3,5,9,15,20],'max_features': ['auto', 'sqrt', 'log2']}

            }
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                            list(model_report.values()).index(best_model_score)
                            ]
            best_model = models[best_model_name]
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            classification_train_metric = get_classification_metrics(y_true=y_train,y_pred=y_train_pred)
            self.track_mlflow(best_model,classification_train_metric)

            classification_test_metric = get_classification_metrics(y_true=y_test,y_pred=y_test_pred)
            self.track_mlflow(best_model,classification_test_metric)

            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            churnModel = ChurnModel(prepocessor=preprocessor,model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path,churnModel)
            save_object("final_model/model.pkl", best_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model training completed successfully")
            return model_trainer_artifact



        except Exception as e:
            raise ProjectException(e, sys)



    def initiate_model_training(self):
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],train_arr[:,-1],
                test_arr[:,:-1],test_arr[:,-1]
            )

            model_trainer_artifact = self.train_model(X_train,y_train,X_test,y_test)
            return model_trainer_artifact

        except Exception as e:
            raise ProjectException(e, sys)


