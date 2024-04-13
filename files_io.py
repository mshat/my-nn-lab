import pickle
from typing import Tuple

import numpy
import numpy as np

from settings import datasets_folder, trained_nns_folder


def save_in_file(output_filename: str, trained_nn):
    with open(f"{trained_nns_folder}/{output_filename}", 'wb') as f:
        pickle.dump(trained_nn, f)


def load_trained_nn(filename: str):
    with open(f"{trained_nns_folder}/{filename}", 'rb') as f:
        return pickle.load(f)


def load_raw_dataset(filename: str, input_data_key: str, target_values_key: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    with np.load(f"{datasets_folder}/{filename}") as f:
        return f[input_data_key], f[target_values_key]
