import yaml
from project.exception.exception import ProjectException
from project.logging.logger import logging
import os,sys
import numpy as np
import dill
import pickle
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold
from sklearn.metrics import recall_score

def read_yaml_file(filepath: str) -> dict:
    try:
        with open(filepath, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise ProjectException(e, sys)
    
def write_yaml_file(filepath: str,content: object,replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(filepath):
                os.remove(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as file:
            yaml.dump(content, file)
    except Exception as e:
        raise ProjectException(e, sys)
    
def save_numpy_Array(file_path: str, array: np.array) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            np.save(file, array)
    except Exception as e:
        raise ProjectException(e, sys)

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info(f"Saving object to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise ProjectException(e, sys)


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found at {file_path}")
        with open(file_path, 'rb') as file:
            print(file)
            return pickle.load(file)
    except Exception as e:
        raise ProjectException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, 'rb') as file:
            return np.load(file)
    except Exception as e:
        raise ProjectException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    try:
        report = {}
        cv = StratifiedKFold(5)
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            rs = RandomizedSearchCV(model, para, cv=cv)
            rs.fit(X_train, y_train)

            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = recall_score(y_train, y_train_pred)
            test_model_score = recall_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise ProjectException(e, sys)
