import abc
from models import NN, TrainedNN
from abstract_dataset_loader import Dataset


class AbstractTrainer(metaclass=abc.ABCMeta):
    def __init__(self, epochs: int, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs

    @abc.abstractmethod
    def train(self, nn: NN, dataset: Dataset) -> TrainedNN:
        pass

