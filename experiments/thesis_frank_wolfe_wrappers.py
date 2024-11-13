import random
from time import time

import numpy as np
from custom_utilities_methods import *

from utils import *
from xcolumns.frank_wolfe import *
from xcolumns.metrics import *
from xcolumns.types import *
from xcolumns.utils import *
from xcolumns.weighted_prediction import *


def make_frank_wolfe_find_and_predict_wrapper(
    find_classfier_using_fw_func: Callable,
):
    def find_and_predict_using_fw(
        y_proba: Matrix,
        k: int = 5,
        seed: int = 0,
        pred_repeat: int = 1,
        val_y_true: Matrix = None,
        val_y_proba: Matrix = None,
        **kwargs,
    ):
        if val_y_true is None or val_y_proba is None:
            raise ValueError("Validation set is required for Frank-Wolfe method")

        rnd_cls, meta = call_function_with_supported_kwargs(
            find_classfier_using_fw_func,
            val_y_true,
            val_y_proba,
            k,
            **kwargs,
        )
        classifiers_a = rnd_cls.a
        classifiers_b = rnd_cls.b
        classifiers_proba = rnd_cls.p

        print(f"  predicting with randomized classfier {pred_repeat} times")
        y_preds = []

        random.seed(seed)
        # meta['pred_meta'] = []
        meta["pred_time"] = []
        for i in range(pred_repeat):
            start_time = time()
            # y_pred, meta = predict_using_randomized_weighted_classifier(
            y_pred = predict_using_randomized_weighted_classifier(
                y_proba,
                k,
                classifiers_a,
                classifiers_b,
                classifiers_proba,
                seed=random.randint(0, int("0x7FFFFFFF", 16)),
                # return_meta=True,
            )
            y_preds.append(y_pred)
            # meta['pred_meta'].append(meta)
            meta["pred_time"].append(time() - start_time)

        if pred_repeat == 1:
            y_preds = y_preds[0]
            # meta['pred_meta'] = meta['pred_meta'][0]
            meta["pred_time"] = meta["pred_time"][0]

        return y_preds, meta

    return add_kwargs_to_signature(
        find_and_predict_using_fw,
        find_classifier_using_fw,
        skip=["seed"],
    )

    return find_and_predict_using_fw


find_and_predict_for_macro_precision_using_fw = (
    make_frank_wolfe_find_and_predict_wrapper(
        find_classifier_optimizing_macro_precision_using_fw
    )
)

find_and_predict_for_macro_recall_using_fw = make_frank_wolfe_find_and_predict_wrapper(
    find_classifier_optimizing_macro_recall_using_fw
)

find_and_predict_for_macro_f1_score_using_fw = (
    make_frank_wolfe_find_and_predict_wrapper(
        find_classifier_optimizing_macro_f1_score_using_fw
    )
)

find_and_predict_for_macro_jaccard_score_using_fw = (
    make_frank_wolfe_find_and_predict_wrapper(
        find_classifier_optimizing_macro_jaccard_score_using_fw
    )
)

find_and_predict_for_macro_balanced_accuracy_using_fw = (
    make_frank_wolfe_find_and_predict_wrapper(
        find_classifier_optimizing_macro_balanced_accuracy_using_fw
    )
)

find_and_predict_for_macro_gmean_using_fw = make_frank_wolfe_find_and_predict_wrapper(
    find_classifier_optimizing_macro_gmean_using_fw
)

find_and_predict_for_macro_hmean_using_fw = make_frank_wolfe_find_and_predict_wrapper(
    find_classifier_optimizing_macro_hmean_using_fw
)
