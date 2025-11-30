import json
import os
import sys

import numpy as np


def reduce_top_values(arr, min_diff=100):
    """
    Set the top n values of the array to zero.
    """
    arr = np.array(arr)
    if len(arr) == 0:
        return arr
    arr = np.array(arr)
    unique_sorted = np.sort(np.unique(arr))
    max_val = unique_sorted[::-1]
    i = 0
    print(f"original max_val: {max_val[0]}")
    while i + 1 < len(max_val) and max_val[i] - max_val[i + 1] > min_diff:
        i += 1
    max_val = max_val[i]
    top_p = np.percentile(unique_sorted, 99.8)
    print(f"new max_val {i}: {max_val} or {top_p} = {min(max_val, top_p)}")
    max_val = min(max_val, top_p)
    # if 1 < len(max_val):
    #     max_val = max_val[1]
    print(f"new max_val {i}: {max_val}")
    arr = np.where(arr >= max_val, max_val, arr)

    return arr


def reduce_top_values_simple(arr):
    """
    Set the top n values of the array to zero.
    """
    arr = np.array(arr)
    if len(arr) == 0:
        return arr
    arr = np.array(arr)
    unique_sorted = np.sort(np.unique(arr))
    max_val = unique_sorted[::-1]
    i = 0
    print(f"original max_val: {max_val[0]}")
    if len(max_val) > 1:
        max_val = max_val[1]
    print(f"new max_val 1: {max_val}")
    arr = np.where(arr >= max_val, max_val, arr)

    return arr


def extract_weights_and_biases(json_file):
    # Read the JSON file
    with open(json_file) as file:
        data = json.load(file)

    # Extract the last lists from "a" and "b"
    weights = data.get("a", [])
    if len(weights) > 0 and isinstance(weights[-1], list):
        weights = weights[min(2, len(weights) - 1)]

    # weights = reduce_top_values(weights)
    weights = reduce_top_values_simple(weights)

    biases = data.get("b", [])
    if biases is None:
        biases = []
    elif len(biases) > 0 and isinstance(biases[-1], list):
        biases = biases[min(2, len(biases) - 1)]
        # biases = reduce_top_values(biases)

    # Generate output file names
    base_name, _ = os.path.splitext(json_file)
    weights_file = f"{base_name}_weights.txt"
    biases_file = f"{base_name}_biases.txt"

    # Save weights to a file
    with open(weights_file, "w") as file:
        file.write("\n".join(map(str, weights)))
    print(f"Weights saved to {weights_file}")

    # Save biases to a file
    if len(biases) != 0 and "recall" not in json_file:
        assert len(weights) == len(
            biases
        ), "Weights and biases must have the same length."
        with open(biases_file, "w") as file:
            file.write("\n".join(map(str, biases)))
        print(f"Biases saved to {biases_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_bf_star_weights.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    if not os.path.isfile(json_file):
        print(f"Error: File '{json_file}' does not exist.")
        sys.exit(1)

    extract_weights_and_biases(json_file)
