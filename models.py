import numpy as np
from typing import List


class Layer:
    def __init__(self, neurons_count: int):
        self.neurons_count = neurons_count


class NN:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.weights = []
        self.bias = []
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            self.weights.append(np.random.uniform(-0.5, 0.5, (next_layer.neurons_count, layer.neurons_count)))
            self.bias.append(np.zeros((next_layer.neurons_count, 1)))

    def __str__(self) -> str:
        return f"nn_{len(self.layers)}_layers"

    def __repr__(self) -> str:
        return f"nn_{len(self.layers)}_layers"


class TrainedLayer:
    def __init__(self, bias, weights):
        self.bias = bias
        self.weights = weights


class TrainedNN:
    def __init__(self, layers_data: List[TrainedLayer]):
        self.layers_count = len(layers_data)
        self.layers = layers_data


