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
    x, y = make_multilabel_classification(
        n_samples=100000,
        n_features=100,
        n_classes=20,
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

    def clf_predict(clf, x):
        """
        Process the output of a multioutput classifier to get the marginal probabilities of each label (class).
        """
        y_proba = clf.predict_proba(x)
        return np.array(y_proba)[:, :, 1].transpose()

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
