from typing import Tuple
import random

import numpy as np
from numpy import ndarray


def gaussian_uniform_crossover(x_parent: ndarray, y_parent: ndarray) -> Tuple[ndarray, ndarray]:
    x_child = np.zeros(x_parent.shape)
    y_child = np.zeros(y_parent.shape)
    for index, (x, y) in enumerate(zip(x_parent, y_parent)):
        distance = abs(x - y)
        random_number = random.random()
        alpha = np.random.normal(0, 1)
        if random_number <= 0.5:
            new_x = x + alpha * distance / 3
            new_y = y + alpha * distance / 3
        else:
            new_x = y + alpha * distance / 3
            new_y = x + alpha * distance / 3
        x_child[index] = new_x
        y_child[index] = new_y
    return x_child, y_child


x_parent = np.array([[1, 2, 5], [2, 3, 2], [3, 4, 9]])
y_parent = np.array([[4, 15, 8], [4, 2, 1], [1, 9, 10]])
x_child, y_child = gaussian_uniform_crossover(x_parent[0], y_parent[0])
print(x_child)
print(y_child)
