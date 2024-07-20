import sys

import click
import numpy as np
from custom_utilities_methods import *
from napkinxc.datasets import load_libsvm_file
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm, trange

from utils import *


@click.command()
@click.argument("dataset", type=str, required=True)
@click.argument("method", type=str, required=True)
def main(
    dataset,
    method,
):
    print(dataset)

    xmlc_data_load_config = {
        "labels_delimiter": ",",
        "labels_features_delimiter": " ",
        "header": True,
    }

    train_y_true_path = None
    valid_y_true_path = None
    train_y_true = None
    valid_y_true = None

    # Predefined datasets
    multiclass = False
    if "news20" in dataset:
        xmlc_data_load_config["header"] = False
        test_y_true_path = "datasets/news20/news20_test.svm"
        train_y_true_path = "datasets/news20/news20_train.svm"
        multiclass = True

    elif "sensorless" in dataset:
        xmlc_data_load_config["header"] = False
        test_y_true_path = "datasets/sensorless/sensorless_test.txt"
        train_y_true_path = "datasets/sensorless/sensorless_train.txt"
        multiclass = True

    elif "ledgar" in dataset:
        xmlc_data_load_config["header"] = False
        test_y_true_path = "datasets/ledgar/ledgar_test.svm"
        train_y_true_path = "datasets/ledgar/ledgar_train.svm"
        multiclass = True

    elif "cal101" in dataset:
        xmlc_data_load_config["header"] = False
        test_y_true_path = "datasets/FLAT_CAL101/FLAT_CAL101.test"
        train_y_true_path = "datasets/FLAT_CAL101/FLAT_CAL101.train"
        multiclass = True

    elif "cal256" in dataset:
        xmlc_data_load_config["header"] = False
        test_y_true_path = "datasets/FLAT_CAL256/FLAT_CAL256.test"
        train_y_true_path = "datasets/FLAT_CAL256/FLAT_CAL256.train"
        multiclass = True

    elif "youtube_deepwalk" in dataset:
        xmlc_data_load_config["header"] = False
        test_y_true_path = "datasets/youtube_deepwalk/youtube_deepwalk_test.svm"
        train_y_true_path = "datasets/youtube_deepwalk/youtube_deepwalk_train.svm"

    elif "eurlex_lexglue" in dataset:
        xmlc_data_load_config["header"] = False
        test_y_true_path = "datasets/eurlex_lexglue/eurlex_lexglue_test.svm"
        train_y_true_path = "datasets/eurlex_lexglue/eurlex_lexglue_train.svm"

    elif "mediamill" in dataset:
        # mediamill - PLT
        xmlc_data_load_config["header"] = False
        test_y_true_path = "datasets/mediamill/mediamill_test.txt"
        train_y_true_path = "datasets/mediamill/mediamill_train.txt"

    elif "flicker_deepwalk" in dataset:
        xmlc_data_load_config["header"] = False
        test_y_true_path = "datasets/flicker_deepwalk/flicker_deepwalk_test.svm"
        train_y_true_path = "datasets/flicker_deepwalk/flicker_deepwalk_train.svm"

    elif "rcv1x" in dataset:
        # RCV1X - PLT + XMLC repo data
        test_y_true_path = "datasets/rcv1x/rcv1x_test.txt"
        train_y_true_path = "datasets/rcv1x/rcv1x_train.txt"

    elif "eurlex" in dataset:
        # Eurlex - PLT + XMLC repo data
        test_y_true_path = "datasets/eurlex/eurlex_test.txt"
        train_y_true_path = "datasets/eurlex/eurlex_train.txt"

    elif "amazoncat" in dataset:
        # AmazonCat - PLT + XMLC repo data
        test_y_true_path = "datasets/amazonCat/amazonCat_test.txt"
        train_y_true_path = "datasets/amazonCat/amazonCat_train.txt"

    elif "wiki10" in dataset:
        # Wiki10 - PLT + XMLC repo data
        test_y_true_path = "datasets/wiki10/wiki10_test.txt"
        train_y_true_path = "datasets/wiki10/wiki10_train.txt"

    else:
        raise RuntimeError(f"No matching dataset: {dataset}")

    # Create binary files for faster loading
    if multiclass:
        labels_format = "list"

    from sklearn.datasets import load_svmlight_file

    # train_X, train_Y = load_libsvm_file(
    #     train_y_true_path, labels_format="list" if multiclass else "csr_matrix", sort_indices=True
    # )
    # test_X, test_Y = load_libsvm_file(
    #     test_y_true_path, labels_format="list" if multiclass else "csr_matrix", sort_indices=True
    # )

    train_X, train_Y = load_svmlight_file(train_y_true_path)
    test_X, test_Y = load_svmlight_file(test_y_true_path)

    # train_Y = np.array(train_Y)
    # test_Y = np.array(test_Y)

    # print(train_X[0], train_Y[0])
    # print(test_X[0], test_Y[0])

    if train_X.shape != test_X.shape:
        align_dim1(train_X, test_X)

    # if train_Y.shape != test_Y.shape:
    #     align_dim1(train_Y, test_Y)

    # train_X = train_X.toarray()
    # test_X = test_X.toarray()
    # train_Y = train_Y.toarray()
    # test_Y = test_Y.toarray()

    assert train_X.shape[1] == test_X.shape[1]
    # assert train_Y.shape[1] == test_Y.shape[1]
    assert train_X.shape[0] == train_Y.shape[0]
    assert test_X.shape[0] == test_Y.shape[0]

    train_n = train_X.shape[0]
    test_n = test_X.shape[0]
    d = train_X.shape[1]
    # m = train_Y.shape[1]
    m = train_Y.max() + 1
    print(f"train_n: {train_n}, test_n: {test_n}, d: {d}, m: {m}")

    # order = np.arange(n)
    # np.random.shuffle(order)
    # X = X[order]
    # Y = Y[order]

    Y_save_path = f"datasets/sklearn/{dataset}_{method}/"
    os.makedirs(Y_save_path, exist_ok=True)
    save_npz(Y_save_path + "train_Y_true.npz", csr_matrix(train_Y))
    save_npz(Y_save_path + "test_Y_true.npz", csr_matrix(test_Y))

    # print("train_Y", train_Y)

    if isinstance(train_Y, csr_matrix) and multiclass:
        if multiclass:
            train_Y = train_Y.indices
            test_Y = test_Y.indices

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # # Standardize the data
    # scaler = StandardScaler(with_mean=False)
    # train_X = scaler.fit_transform(train_X)
    # test_X = scaler.transform(test_X)

    if method != "lr":
        raise RuntimeError(f"No matching method: {method}")

    # Train the model
    print("Training the model ...")
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        verbose=1,
        multi_class="multinomial" if multiclass else "ovr",
    )
    model.fit(train_X, train_Y)
    train_Y_pred = model.predict_proba(train_X)
    test_Y_pred = model.predict_proba(test_X)
    # print(test_Y_pred)

    top_1 = test_Y_pred.argmax(axis=1)
    accuracy = (top_1 == test_Y).mean()
    print(f"Accuracy: {accuracy}")

    # Save the predictions
    print("Saving predictions ...")
    save_npz(Y_save_path + "train_Y_pred.npz", csr_matrix(train_Y_pred))
    save_npz(Y_save_path + "test_Y_pred.npz", csr_matrix(test_Y_pred))


if __name__ == "__main__":
    main()
