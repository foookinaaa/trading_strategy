from typing import List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_metrics_for_clf(
    y_true: pd.Series, y_pred: pd.Series, conf_matrx_labels: List[str] = None
) -> tuple:
    """
    Get accuracy, roc_auc, precision, recall, confusion_matrix for binary classification

    :param y_true: array with true target
    :param y_pred: array with predictions
    :param conf_matrx_labels: labels for confusion_matrix
    :return: all metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=conf_matrx_labels)
    return accuracy, roc_auc, precision, recall, cm
