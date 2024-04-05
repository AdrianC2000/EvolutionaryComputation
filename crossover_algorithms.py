from typing import List, Tuple, Optional

import numpy as np
from numpy import ndarray


class CrossoverAlgorithms:

    def __init__(self, crossover_method: str, binary_chain_length: int, crossover_probability: float):
        self.__METHODS = {
            "single-point": self._single_point_crossover,
            "mssx": self._mssx_crossover
        }
        self.__crossover_method = self.__METHODS[crossover_method]
        self.__binary_chain_length = binary_chain_length
        self.__crossover_point = int(binary_chain_length / 2)
        self.__crossover_probability = crossover_probability

    def perform_crossover(self, selected_parents_encoded: ndarray) -> ndarray:
        np.random.shuffle(selected_parents_encoded)
        descendants = np.empty_like(selected_parents_encoded)

        for i in range(0, len(selected_parents_encoded), 2):

            if i + 1 >= len(selected_parents_encoded):
                descendants[i] = selected_parents_encoded[i]
                break

            parent_a = selected_parents_encoded[i]
            parent_b = selected_parents_encoded[i + 1]

            if np.random.rand() > self.__crossover_probability:
                descendants[i], descendants[i + 1] = parent_a, parent_b
            else:
                children = self.__crossover_method(parent_a, parent_b)
                descendants[i], descendants[i + 1] = children

        return descendants

    def _single_point_crossover(self, parent_a: ndarray, parent_b: ndarray) -> Tuple[ndarray, ndarray]:
        parent_a_combined = self._combine_parent(parent_a)
        parent_b_combined = self._combine_parent(parent_b)

        descendant_1 = parent_a_combined[:self.__crossover_point] + parent_b_combined[self.__crossover_point:]
        descendant_2 = parent_b_combined[:self.__crossover_point] + parent_a_combined[self.__crossover_point:]

        return self._map_descendant(descendant_1), self._map_descendant(descendant_2)

    def _mssx_crossover(self, parent_a: ndarray, parent_b: ndarray) -> Tuple[ndarray, ndarray]:
        parent_a_combined = np.array(list(self._combine_parent(parent_a)), dtype=int)
        parent_b_combined = np.array(list(self._combine_parent(parent_b)), dtype=int)

        sexes_count = 2
        n = len(parent_a_combined)
        offsprings = np.zeros((2, n)).astype(int)
        rng = np.random.default_rng()

        parents = np.vstack((parent_a_combined, parent_b_combined))

        for offspring in range(offsprings.shape[0]):
            for i in range(n):
                random_sex = rng.integers(sexes_count)
                offsprings[offspring][i] = parents[random_sex][i]

        descendant_1 = self._combine_parent(offsprings[0])
        descendant_2 = self._combine_parent(offsprings[1])
        return self._map_descendant(descendant_1), self._map_descendant(descendant_2)

    @staticmethod
    def _combine_parent(parent: ndarray) -> str:
        return "".join(str(element) for element in parent)

    def _map_descendant(self, descendant: str) -> ndarray:
        return np.array([descendant[i:i + self.__binary_chain_length]
                         for i in range(0, len(descendant), self.__binary_chain_length)])
