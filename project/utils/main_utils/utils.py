import yaml
from project.exception.exception import ProjectException
from project.logging.logger import logging
import os,sys
import numpy as np
import dill
import pickle

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
