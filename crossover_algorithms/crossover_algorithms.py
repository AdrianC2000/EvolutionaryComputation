import numpy as np
from numpy import ndarray


class CrossoverAlgorithms:
    def __init__(self, crossover_method: str, crossover_probability: float, methods: dict):
        self.__crossover_method = methods[crossover_method]
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
