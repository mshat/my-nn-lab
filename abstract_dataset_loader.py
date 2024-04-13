import abc

from files_io import load_raw_dataset


class Dataset:
    def __init__(self, inputs, targets):
        assert len(inputs) == len(targets)

        self.inputs = inputs
        self.targets = targets
        self.len = len(inputs)


class AbstractDatasetLoader(metaclass=abc.ABCMeta):
    def load_dataset(self, filename: str, input_data_key: str, target_values_key: str) -> Dataset:
        inputs, targets = load_raw_dataset(filename, input_data_key, target_values_key)
        return self.normalize_dataset(Dataset(inputs, targets))

    @abc.abstractmethod
    def normalize_dataset(self, dataset: Dataset) -> Dataset:
        pass
