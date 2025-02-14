from project.entity.artifact_entity import ClassificationMetricArtifact
from project.exception.exception import ProjectException
from sklearn.metrics import precision_score, recall_score, f1_score
import sys

def get_classification_metrics(y_true, y_pred)-> ClassificationMetricArtifact:
    try:
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        return ClassificationMetricArtifact(f1, precision, recall)
    except Exception as e:
        raise ProjectException(e, sys)