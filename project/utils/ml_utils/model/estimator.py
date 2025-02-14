from project.constants.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME

import os,sys

from project.logging.logger import logging
from project.exception.exception import ProjectException


class ChurnModel:
    def __init__(self, model,prepocessor):
        try:
            self.model = model
            self.prepocessor = prepocessor
        except Exception as e:
            raise ProjectException(e, sys)
    def predict(self, x):
        try:
            x = self.prepocessor.transform(x)
            y_hat = self.model.predict(x)
            return y_hat
        except Exception as e:
            raise ProjectException(e, sys)
