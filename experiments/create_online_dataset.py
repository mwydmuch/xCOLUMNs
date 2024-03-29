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
@click.option("-s", "--seed", type=int, required=True)
def main(
    dataset,
    method,
    seed,
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
    if "youtube_deepwalk" in dataset:
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
    train_X, train_Y = load_libsvm_file(
        train_y_true_path, labels_format="csr_matrix", sort_indices=True
    )
    test_X, test_Y = load_libsvm_file(
        test_y_true_path, labels_format="csr_matrix", sort_indices=True
    )

    # print(train_X[0], train_Y[0])
    # print(test_X[0], test_Y[0])

    if train_X.shape != test_X.shape:
        align_dim1(train_X, test_X)

    if train_Y.shape != test_Y.shape:
        align_dim1(train_Y, test_Y)

    # train_X = train_X.toarray()
    # test_X = test_X.toarray()
    # train_Y = train_Y.toarray()
    # test_Y = test_Y.toarray()

    assert train_X.shape[1] == test_X.shape[1]
    assert train_Y.shape[1] == test_Y.shape[1]
    assert train_X.shape[0] == train_Y.shape[0]
    assert test_X.shape[0] == test_Y.shape[0]

    X = vstack([train_X, test_X], format="csr")
    Y = vstack([train_Y, test_Y], format="csr")

    n = X.shape[0]
    d = X.shape[1]
    m = Y.shape[1]
    print(f"n: {n}, d: {d}, m: {m}")

    # order = np.arange(n)
    # np.random.shuffle(order)
    # X = X[order]
    # Y = Y[order]

    Y_save_path = f"datasets/online/{dataset}/{method}_s={seed}/"
    os.makedirs(Y_save_path, exist_ok=True)
    save_npz(Y_save_path + "Y_true.npz", csr_matrix(Y))

    if method != "linear_adam":
        raise RuntimeError(f"No matching method: {method}")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Define a simple linear model with a single output
    class LinearModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            x = self.linear(x)
            x = torch.sigmoid(x)  # Sigmoid activation to output probabilities
            return x

    # Initialize the model
    model = LinearModel(input_dim=d, output_dim=m)

    # Define the loss function (Binary Cross Entropy)
    criterion = nn.BCELoss()

    # Define the optimizer (Stochastic Gradient Descent)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.00001, lr=0.005)

    # Sample data (x, y) where x is the input and y is the label (0 or 1)
    # X = torch.tensor(X, dtype=torch.float32)
    # Y = torch.tensor(Y, dtype=torch.float32)

    loss_sum = 0
    l_i = 0
    epochs = 1
    all_outputs = []
    p_at_1 = 0

    # Training loop
    for _ in range(epochs):
        order = np.arange(n)
        np.random.shuffle(order)
        X = X[order]
        Y = Y[order]
        save_npz(Y_save_path + "Y_true.npz", csr_matrix(Y))
        all_outputs = []

        torch_X = torch.sparse_csr_tensor(
            torch.tensor(X.indptr, dtype=torch.int32),
            torch.tensor(X.indices, dtype=torch.int32),
            torch.tensor(X.data, dtype=torch.float32),
            torch.Size(X.shape),
        )

        # torch_Y = torch.sparse_csr_tensor(
        #     torch.tensor(Y.indptr, dtype=torch.int32),
        #     torch.tensor(Y.indices, dtype=torch.int32),
        #     torch.tensor(Y.data, dtype=torch.float32),
        #     torch.Size(Y.shape))

        torch_Y = torch.tensor(Y.toarray(), dtype=torch.float32)

        t = trange(n, desc="Loss")
        for i in t:
            # Forward pass: Compute predicted y by passing x to the model
            optimizer.zero_grad()
            # x = torch.tensor(X[i].toarray(), dtype=torch.float32)
            x = torch_X[i]
            outputs = model(x)
            p_at_1 += Y[i, torch.argmax(outputs)]

            # Compute and print loss
            # y = torch.tensor(Y[i].toarray(), dtype=torch.float32)
            y = torch_Y[i]
            loss = criterion(outputs, y)
            loss_sum += loss.item()

            # Zero gradients, perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()

            l_i += 1
            t.set_description(f"Loss: {loss_sum / l_i}, P@1: {p_at_1 / l_i}")
            all_outputs.append(outputs.detach())

        # Save the predictions
        print("Saving predictions ...")
        all_outputs = torch.stack(all_outputs)
        all_outputs = all_outputs.detach().numpy()
        save_npz(Y_save_path + "Y_pred.npz", csr_matrix(all_outputs))


if __name__ == "__main__":
    main()
