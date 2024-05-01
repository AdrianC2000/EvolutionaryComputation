from abc import ABC
from typing import Tuple, List

import numpy as np
from numpy import ndarray

from fitness_function import FitnessFunction
from selection_algorithms import SelectionAlgorithms


class GeneticAlgorithm(ABC):

    def __init__(self, bounds: Tuple[int, int], variables_number: int, selection_method: str,
                 elitism_ratio: float, is_min_searched: bool = False, tournaments_count: int = 3,
                 fraction_selected: float = 0.34):
        self.__bounds = bounds
        self._variables_number = variables_number
        self._selection_algorithms = SelectionAlgorithms(selection_method)
        self.__elitism_ratio = elitism_ratio
        self._is_min_searched = is_min_searched
        self._tournaments_count = tournaments_count
        self._fraction_selected = fraction_selected

        self._fitness_history = []
        self._average_fitness_history = []
        self._std_dev_fitness_history = []

    def find_best_solution(self, population_size: int, epochs_number: int) -> Tuple[ndarray, float]:
        pass

    def _initialize_population(self, population_size: int, variables_number: int) -> ndarray:
        return np.random.uniform(self.__bounds[0], self.__bounds[1], size=(population_size, variables_number))

    @staticmethod
    def _get_best_individual(population: ndarray) -> Tuple[ndarray, float]:
        fitnesses = [FitnessFunction.fitness_function(individual) for individual in population]
        best_index = np.argmax(fitnesses)
        return population[best_index], fitnesses[best_index]

    def _replace_population(self, population: ndarray, children: ndarray) -> ndarray:
        fitness = [FitnessFunction.fitness_function(individual) for individual in population]
        elites_count = int(self.__elitism_ratio * len(population))

        extremes = self._get_extremes(elites_count, fitness)
        elites_indices = self._get_elites_indices(extremes, fitness)
        available_indices = np.setdiff1d(np.arange(population.shape[0]), elites_indices)

        children, replacement_indices = self._get_replacement_indices(available_indices, children)

        population[replacement_indices] = children
        return population

    def _get_extremes(self, elites_count: int, fitness: List[float]) -> ndarray:
        if self._is_min_searched:
            return np.partition(fitness, elites_count)[:elites_count]
        return np.partition(fitness, -elites_count)[-elites_count:]

    @staticmethod
    def _get_elites_indices(extremes: ndarray, fitness: List[float]) -> ndarray:
        elites_indices = np.where(np.in1d(fitness, extremes))[0]
        if len(elites_indices) > len(extremes):
            elites_indices = np.random.choice(elites_indices, size=len(extremes), replace=False)
        return elites_indices

    @staticmethod
    def _get_replacement_indices(available_indices: ndarray, children: ndarray) -> Tuple[ndarray, ndarray]:
        if children.shape[0] > len(available_indices):
            children_indices_to_remove = np.random.choice(children.shape[0],
                                                          size=children.shape[0] - len(available_indices),
                                                          replace=False)
            children = np.delete(children, children_indices_to_remove, axis=0)
            replacement_indices = available_indices
        else:
            replacement_indices = np.random.choice(available_indices, size=children.shape[0], replace=False)
        return children, replacement_indices
