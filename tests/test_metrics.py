import numpy as np

from xcolumns.confusion_matrix import calculate_confusion_matrix
from xcolumns.metrics import (
    binary_balanced_accuracy_on_conf_matrix,
    binary_f1_score_on_conf_matrix,
    binary_fbeta_score_on_conf_matrix,
    binary_gmean_on_conf_matrix,
    binary_hmean_on_conf_matrix,
    binary_jaccard_score_on_conf_matrix,
    binary_precision_on_conf_matrix,
    binary_recall_on_conf_matrix,
)
from xcolumns.weighted_prediction import predict_top_k


def test_binary_metrics_on_conf_matrix(generated_test_data):
    y_test = generated_test_data["y_test"]
    y_proba_test = generated_test_data["y_proba_test"]
    k = 3

    # Run predict_top_k to get baseline classifier and initial prediction
    top_k_y_pred = predict_top_k(y_proba_test, k)
    top_k_C = calculate_confusion_matrix(
        y_test, top_k_y_pred, normalize=False, skip_tn=False
    )

    for metric in [
        binary_balanced_accuracy_on_conf_matrix,
        binary_fbeta_score_on_conf_matrix,
        binary_f1_score_on_conf_matrix,
        binary_precision_on_conf_matrix,
        binary_recall_on_conf_matrix,
        binary_jaccard_score_on_conf_matrix,
        binary_gmean_on_conf_matrix,
        binary_hmean_on_conf_matrix,
    ]:
        print(f"testing {metric.__name__}")
        result = metric(*top_k_C)
        print(f"  result={result}")
        assert isinstance(result, np.ndarray)
        assert result.shape == (y_proba_test.shape[1],)
        assert (0 <= result).all() and (result <= 1).all()
