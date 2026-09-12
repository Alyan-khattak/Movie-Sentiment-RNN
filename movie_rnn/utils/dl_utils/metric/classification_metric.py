# ═══════════════════════════════════════════════════════════════════
# movie_rnn/utils/dl_utils/metric/classification_metric.py
# ═══════════════════════════════════════════════════════════════════
# Model evaluation metrics — accuracy, precision, recall, F1
# ═══════════════════════════════════════════════════════════════════

import sys
import logging
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from movie_rnn.exception.exception import MovieSentimentException


@dataclass
class ClassificationMetricArtifact:
    accuracy:  float
    precision: float
    recall:    float
    f1_score:  float


def get_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> ClassificationMetricArtifact:
    """
    y_true : actual labels  [0, 1, 1, 0, ...]
    y_pred : predicted labels [0, 1, 0, 0, ...]  ← already thresholded at 0.5

    Returns ClassificationMetricArtifact
    """
    try:
        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall    = recall_score(y_true, y_pred)
        f1        = f1_score(y_true, y_pred)
        # DRY RUN:
        # y_true = [1, 0, 1, 1, 0]
        # y_pred = [1, 0, 1, 0, 0]
        # accuracy  = 4/5 = 0.80
        # precision = 2/2 = 1.0
        # recall    = 2/3 = 0.67
        # f1        = 0.80

        logging.info(
            f"Metrics | acc: {accuracy:.4f} | "
            f"prec: {precision:.4f} | "
            f"rec: {recall:.4f} | "
            f"f1: {f1:.4f}"
        )

        return ClassificationMetricArtifact(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
        )

    except Exception as e:
        raise MovieSentimentException(e, sys)