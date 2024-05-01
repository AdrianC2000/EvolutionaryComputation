from typing import Tuple

import numpy as np
from numpy import ndarray

from crossover_algorithms.crossover_algorithms import CrossoverAlgorithms


class CrossoverAlgorithmsReal(CrossoverAlgorithms):

    def __init__(self, crossover_method: str, crossover_probability: float, bounds: Tuple[int, int]):
        __METHODS = {
            "arithmetical": self._arithmetical_crossover,
            "linear": self._linear_crossover,
            "blend_alpha": self._blend_alpha_crossover,
            "blend_alpha_and_beta": self._blend_alpha_and_beta_crossover,
            "average": self._average_crossover,
            "differential_evolution": self._differential_evolution_crossover,  # BARY TO TWOJE
            "unfair_average": self._unfair_average_crossover,  # PIOTER TO TWOJE
            "gaussian_uniform": self._gaussian_uniform_crossover,
        }
        super().__init__(crossover_method, crossover_probability, __METHODS)
        self.__bounds = bounds

    @staticmethod
    def _arithmetical_crossover(parent_a: ndarray, parent_b: ndarray) -> Tuple[ndarray, ndarray]:
        alpha = np.random.rand()
        child_1 = alpha * parent_a + (1 - alpha) * parent_b
        child_2 = alpha * parent_b + (1 - alpha) * parent_a

        return child_1, child_2

    def _linear_crossover(self, parent_a: ndarray, parent_b: ndarray) -> Tuple[ndarray, ndarray]:
        # TODO
        return parent_a, parent_b

    def _blend_alpha_crossover(self, parent_a: ndarray, parent_b: ndarray) -> Tuple[ndarray, ndarray]:
        # TODO
        return parent_a, parent_b

    def _blend_alpha_and_beta_crossover(self, parent_a: ndarray, parent_b: ndarray) -> Tuple[ndarray, ndarray]:
        # TODO
        return parent_a, parent_b

    def _average_crossover(self, parent_a: ndarray, parent_b: ndarray) -> Tuple[ndarray, ndarray]:
        # TODO
        return parent_a, parent_b

    def _differential_evolution_crossover(self, parent_a: ndarray, parent_b: ndarray) -> Tuple[ndarray, ndarray]:
        # TODO
        return parent_a, parent_b

    def _unfair_average_crossover(self, parent_a: ndarray, parent_b: ndarray) -> Tuple[ndarray, ndarray]:
        # TODO
        return parent_a, parent_b

    def _gaussian_uniform_crossover(self, x_parent: ndarray, y_parent: ndarray) -> Tuple[ndarray, ndarray]:
        x_child = np.copy(x_parent)
        y_child = np.copy(y_parent)
        for index, (x, y) in enumerate(zip(x_parent, y_parent)):
            distance = abs(x - y)
            random_number = np.random.rand()
            alpha = np.random.normal(0, 1)
            if random_number <= 0.5:
                new_x = x + alpha * distance / 3
                new_y = y + alpha * distance / 3
            else:
                new_x = y + alpha * distance / 3
                new_y = x + alpha * distance / 3
            if self.__bounds[0] <= new_x <= self.__bounds[1]:
                x_child[index] = new_x
            if self.__bounds[0] <= new_y <= self.__bounds[1]:
                y_child[index] = new_y
        return x_child, y_child
