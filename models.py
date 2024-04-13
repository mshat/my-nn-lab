import numpy as np
from typing import List


class NeuralLayer:
    def __init__(self, neurons_count: int):
        self.neurons_count = neurons_count

    def __str__(self) -> str:
        return f"{self.__class__.__name__} ({self.neurons_count})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.neurons_count})"


class NN:
    def __init__(self, layers: List[NeuralLayer]):
        self.layers = layers
        self.weights = []
        self.bias = []
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            self.weights.append(np.random.uniform(-0.5, 0.5, (next_layer.neurons_count, layer.neurons_count)))
            self.bias.append(np.zeros((next_layer.neurons_count, 1)))

    def __str__(self) -> str:
        return f"{self.__class__.__name__} ({len(self.layers)} layers)"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({len(self.layers)} layers)"


class LinearLayer:
    def __init__(self, bias, weights):
        self.bias = bias
        self.weights = weights

    @property
    def input_dim_count(self) -> int:
        return self.weights.shape[1]

    @property
    def output_dim_count(self) -> int:
        return self.weights.shape[0]

    def __str__(self) -> str:
        return f"{self.__class__.__name__} ({self.output_dim_count}, {self.input_dim_count})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.output_dim_count}, {self.input_dim_count})"


class TrainedNN:
    def __init__(self, layers: List[LinearLayer]):
        self.layers = layers

    @property
    def layers_count(self):
        return len(self.layers)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} ({self.layers_count} layers)"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.layers_count} layers)"



