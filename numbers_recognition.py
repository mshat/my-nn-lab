import math
import numpy as np
import matplotlib.pyplot as plt

from tools import sigmoid, predict_sample
from models import Layer, NN, TrainedLayer, TrainedNN
from abstracttrainer import AbstractTrainer
from abstract_dataset_loader import AbstractDatasetLoader, Dataset
from abstract_laboratory import AbstractLaboratory
from synapse_visualizer import visualize_synapses, visualize_synapse


class NumRecDatasetLoader(AbstractDatasetLoader):
    def normalize_dataset(self, dataset: Dataset) -> Dataset:
        # Загрузка и нормализация входных данных. Значения цветов пикселей 0-255 преобразуем в числа в диапазоне от 0 до 1
        inputs = dataset.inputs.astype("float32") / 255

        # преобразуем матрицу массивов цветов пикселей в один одномерный массив длиной n
        inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1] * inputs.shape[2]))

        # загружаем целевые значения (например, цифры, соответствующие рисункам цифр)
        targets = dataset.targets

        # преобразуем целевые значения в удобный формат - вектор с началом в начале координат
        targets = np.eye(10)[targets]

        return Dataset(inputs, targets)


class NumRecTrainer(AbstractTrainer):
    def train(self, nn: NN, dataset: Dataset) -> TrainedNN:
        parameter_count = dataset.inputs.shape[0]

        # Веса и смещения для первого скрытого слоя
        e_loss, e_correct = 0, 0

        for epoch in range(self.epochs):
            print(f"Epoch №{epoch}")

            output = None
            for x, y in zip(dataset.inputs, dataset.targets):
                x = np.reshape(x, (-1, 1))  # транспонирование каждого массива (объекта) из 1хN в Nx1
                y = np.reshape(y, (-1, 1))  # транспонирование каждого массива (объекта) из 1хN в Nx1

                # передача данных из входного в скрытый слой (прямое распространение)
                # нормализация функцией активации сигмоид
                layer_data = x
                layer_datas = [x]
                for weights, bias in zip(nn.weights, nn.bias):
                    layer_data = sigmoid(weights @ layer_data + bias)
                    layer_datas.append(layer_data)

                output = layer_data

                # расчёты потерь / ошибок с помощью DME
                error = output - y
                e_loss += 1 / len(output) * np.sum(error ** 2, axis=0)
                # e_loss += np.sum(error ** 2, axis=0)
                e_correct += int(np.argmax(output) == np.argmax(y))

                # обучение, алгоритм Backpropagation
                # Обратное распространение
                # Обновление весов и смещений

                delta = error * output * (1 - output)
                nn.weights[-1] -= self.learning_rate * (delta @ layer_datas[-2].T)
                nn.bias[-1] -= self.learning_rate * np.sum(delta, axis=1, keepdims=True)

                for i in range(len(nn.weights) - 1, -1, -1):
                    current_layer_data = layer_datas[i]
                    next_layer_delta = delta
                    delta = (nn.weights[i].T @ next_layer_delta) * current_layer_data * (1 - current_layer_data)

                    nn.weights[i] -= self.learning_rate * (next_layer_delta @ current_layer_data.T)
                    nn.bias[i] -= self.learning_rate * np.sum(next_layer_delta, axis=1, keepdims=True)

            print(f"Loss: {round((e_loss[0] / parameter_count) * 100, 3)}%")
            print(f"Accuracy: {round((e_correct / parameter_count) * 100, 3)}%")
            e_loss = 0
            e_correct = 0
        return TrainedNN([TrainedLayer(bias, weights) for bias, weights in zip(nn.bias, nn.weights)])


class NumRecLaboratory(AbstractLaboratory):
    dataset_filename = "mnist.npz"
    dataset_loader = NumRecDatasetLoader
    dataset_input_values_key = "x_train"
    dataset_target_values_key = "y_train"

    def custom_test(self, nn: TrainedNN):
        test_image = plt.imread("custom.jpg", format="jpeg")

        # Grayscale + Unit RGB + inverse colors
        gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        test_image = 1 - (gray(test_image).astype("float32") / 255)

        # Reshape
        test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

        # Predict
        image = np.reshape(test_image, (-1, 1))

        res = predict_sample(image, nn)

        plt.imshow(test_image.reshape(28, 28), cmap="Greys")
        plt.title(f"NN suggests the CUSTOM number is: {res}")
        plt.show()

    def show_layer_synapses(self, nn: TrainedNN, layer_index: int, limit=100) -> None:
        last_layer = nn.layers[layer_index]
        n_synapses = last_layer.weights.shape[0]
        n_parameters = last_layer.weights.shape[1]

        # можно ли представить синапсы в виде квадратного массива
        if (frame_height_width := math.isqrt(n_parameters)) ** 2 != n_parameters:
            return

        neurons = [
                      np.reshape(last_layer.weights[i] + last_layer.bias[i], (frame_height_width, frame_height_width))
                      for i in range(n_synapses)
                  ][:limit]
        visualize_synapses(neurons)

    def show_synapse(self, nn: TrainedNN, layer_index: int, sinaps_index: int) -> None:
        layer = nn.layers[layer_index]
        n_parameters = layer.weights.shape[1]

        # можно ли представить синапсы в виде квадратного массива
        if (frame_height_width := math.isqrt(n_parameters)) ** 2 != n_parameters:
            return

        neuron = np.reshape(
            layer.weights[sinaps_index] + layer.bias[sinaps_index], (frame_height_width, frame_height_width)
        )
        visualize_synapse(neuron)
