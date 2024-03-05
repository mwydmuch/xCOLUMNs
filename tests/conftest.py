# Create data for testing using sklearn

import numpy as np
import pytest
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier


@pytest.fixture(autouse=True, scope="session")
def generated_test_data():
    seed = 2024
    max_top_k = 5
    x, y = make_multilabel_classification(
        n_samples=100000,
        n_features=100,
        n_classes=50,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        sparse=False,
        return_indicator="dense",
        return_distributions=False,
        random_state=seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.3, random_state=seed
    )
    clf = MultiOutputClassifier(LogisticRegression()).fit(x_train, y_train)

    def clf_predict(clf, x, sparsfy=False):
        """
        Process the output of a multioutput classifier to get the marginal probabilities of each label (class).
        """
        y_proba = clf.predict_proba(x)

        # Convert the output of MultiOutputClassifier that contains the marginal probabilities for both P(y=0) and P(y=1) to just a matrix of P(y=1)
        y_proba = np.array(y_proba)[:, :, 1].transpose()

        # Sparify the matrix
        if sparsfy:
            top_k_thr = -np.partition(-y_proba, max_top_k, axis=1)[:, max_top_k]
            top_k_thr = top_k_thr.reshape((-1,))
            top_k_thr[top_k_thr >= 0.1] = 0.1
            y_proba[y_proba < top_k_thr[:, None]] = 0
            assert ((y_proba > 0).sum(axis=1) >= max_top_k).all()

        return y_proba

    y_proba_train = clf_predict(clf, x_train)
    y_proba_val = clf_predict(clf, x_val)
    y_proba_test = clf_predict(clf, x_test)

    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        y_proba_train,
        y_proba_val,
        y_proba_test,
    )
