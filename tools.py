import numpy as np
from models import TrainedNN


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_sample(sample, nn: TrainedNN) -> int | float:
    """
    Подаёт на вход обученной нейросети семпл и возвращает предсказание на его основе
    """
    layer_data = sample
    for layer in nn.layers:
        layer_data = sigmoid((layer.bias + layer.weights @ layer_data))
    return layer_data.argmax()
