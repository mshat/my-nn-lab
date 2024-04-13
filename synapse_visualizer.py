import math
from typing import List

import numpy as np

import matplotlib.pyplot as plt


def visualize_synapse(synapse: np.ndarray) -> None:
    """
    Визуализирует NxN ndarray, где цвет каждого пикселя зависит от значения элемента:
    Отрицательные значения отображаются синим цветом, положительные - красным.
    """
    # Создание изображения с использованием цветовой карты 'coolwarm'
    plt.imshow(synapse, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()  # Добавление шкалы для наглядности
    plt.show()


def visualize_synapses(synapses: List[np.ndarray]) -> None:
    """
    Визуализирует список NxN ndarray в квадратном гриде, где каждый элемент отображает один массив.
    Цвета пикселей зависят от значений в массивах: отрицательные - синим, положительные - красным.

    Параметры:
        arrays (list of ndarray): Список двумерных массивов размера NxN с элементами от -1 до 1.
    """
    n = len(synapses)  # количество массивов
    grid_size = math.ceil(math.sqrt(n))  # размерность сетки для субплотов

    # Создаем грид для субплотов
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))

    # Отображаем каждый массив
    for i, ax in enumerate(axs.flat):
        if i < n:
            arr = synapses[i]
            # Отображаем массив
            im = ax.imshow(arr, cmap='coolwarm', interpolation='nearest')
            ax.set_title(f"Array {i}")
        ax.axis('off')  # выключаем оси, где не нужно

    # Добавляем общий colorbar
    fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.show()
