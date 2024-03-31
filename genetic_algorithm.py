from typing import Tuple

import numpy as np
from numpy import ndarray

from binary_encoder import BinaryEncoder
from fitness_function import FitnessFunction
from selection_algorithms import SelectionAlgorithms


class GeneticAlgorithm:

    def __init__(self, precision: int, bounds: Tuple[int, int], selection_method: str):
        self.__bounds = bounds
        self.__binary_encoder = BinaryEncoder(precision, bounds[0], bounds[1])
        self.__selection_algorithms = SelectionAlgorithms(selection_method)

    def find_best_solution(self, population_size: int, variables_number: int, epochs_number: int):
        population = self._initialize_population(population_size, variables_number)
        best_fitness, best_individual = 0, None

        for _ in range(epochs_number):
            new_best_individual, new_best_fitness = self._get_best_individual(population)
            if new_best_fitness > best_fitness:
                best_fitness = new_best_fitness
                best_individual = new_best_individual

            selected_parents_population = self.__selection_algorithms.select_parents(population)
            selected_parents_encoded = self.__binary_encoder.encode_population(selected_parents_population)

    def _initialize_population(self, population_size: int, variables_number: int) -> ndarray:
        return np.random.uniform(self.__bounds[0], self.__bounds[1], size=(population_size, variables_number))

    @staticmethod
    def _get_best_individual(population: ndarray) -> Tuple[ndarray, float]:
        fitnesses = [FitnessFunction.fitness_function(individual) for individual in population]
        best_index = np.argmax(fitnesses)
        return population[best_index], fitnesses[best_index]
