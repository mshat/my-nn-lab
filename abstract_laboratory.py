import abc

import numpy as np
from models import Layer, NN, TrainedLayer, TrainedNN

from files_io import save_in_file, load_trained_nn


from abstracttrainer import AbstractTrainer
from abstract_dataset_loader import AbstractDatasetLoader, Dataset
from tools import predict_sample


class AbstractLaboratory(metaclass=abc.ABCMeta):
    dataset_filename: str
    dataset_loader: type(AbstractDatasetLoader)
    dataset_input_values_key: str
    dataset_target_values_key: str

    def __init__(self, trainer: AbstractTrainer):
        self.trainer = trainer

    def load_dataset(self):
        return self.dataset_loader().load_dataset(
            self.dataset_filename, self.dataset_input_values_key, self.dataset_target_values_key
        )

    def load_first_n_samples(self, n: int):
        dataset = self.dataset_loader().load_dataset(
            self.dataset_filename, self.dataset_input_values_key, self.dataset_target_values_key
        )
        return Dataset(dataset.inputs[:n], dataset.targets[:n])

    def load_last_n_samples(self, n: int):
        dataset = self.dataset_loader().load_dataset(
            self.dataset_filename, self.dataset_input_values_key, self.dataset_target_values_key
        )
        return Dataset(dataset.inputs[dataset.len - n:], dataset.targets[dataset.len - n:])

    def train(self, nn: NN, dataset: Dataset, output_filename: str, replace_existing=False) -> TrainedNN:
        try:
            existing_nn = load_trained_nn(output_filename)
        except FileNotFoundError:
            existing_nn = None
        if existing_nn and not replace_existing:
            return existing_nn

        trained_nn = self.trainer.train(nn, dataset)

        save_in_file(output_filename, trained_nn)
        return trained_nn

    def test_nn(self, test_dataset: Dataset, nn: TrainedNN):
        success_count = 0

        for x, y in zip(test_dataset.inputs, test_dataset.targets):
            sample = x.reshape(x.shape[0], 1)
            res = predict_sample(sample, nn)
            success_count += int(res == np.where(y == 1.0)[0][0])

        print(f"Тест на {test_dataset.len} сэмплов")
        print(f"Успешно распознано {round(success_count * 100 / test_dataset.len, 2)}%")

    @abc.abstractmethod
    def custom_test(self, nn: TrainedNN):
        pass
