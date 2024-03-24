import random
from typing import Tuple, List

import numpy as np
from numpy import ndarray

from binary_encoder import BinaryEncoder


class GeneticAlgorithm:

    def __init__(self, precision: int, bounds: Tuple[int, int]):
        self.__bounds = bounds
        self.__binary_encoder = BinaryEncoder(precision, bounds[0], bounds[1])

    def find_best_solution(self, population_size: int, variables_number: int, epochs_number: int):
        initial_population = self._initialize_population(population_size, variables_number)
        initial_population_encoded = self._encode_population(initial_population)


    def _initialize_population(self, population_size: int, variables_number: int) -> ndarray:
        return np.random.uniform(self.__bounds[0], self.__bounds[1], size=(population_size, variables_number))

    def _encode_population(self, population: ndarray) -> ndarray:
        return np.apply_along_axis(self._encode_individual, 1, population)

    def _encode_individual(self, individual: ndarray) -> ndarray:
        vectorized_method = np.vectorize(self.__binary_encoder.encode_to_binary)
        return vectorized_method(individual)
