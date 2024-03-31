import numpy as np
from numpy import ndarray

from fitness_function import FitnessFunction


class SelectionAlgorithms:

    def __init__(self, selection_method: str):
        self.__METHODS = {
            "tournament": self._tournament_selection
        }
        self.__selected_method = self.__METHODS[selection_method]

    def select_parents(self, population: ndarray) -> ndarray:
        return self.__selected_method(population)

    @staticmethod
    def _tournament_selection(population: ndarray) -> ndarray:
        selected_parents = np.empty((population.shape[0], population.shape[1]))

        for i in range(len(population)):
            tournament_indices = np.random.choice(len(population), size=3, replace=False)
            tournament_individuals = population[tournament_indices]

            tournament_fitness = [FitnessFunction.fitness_function(individual) for individual in tournament_individuals]
            parent_index = np.argmax(tournament_fitness)
            selected_parents[i] = tournament_individuals[parent_index]

        return selected_parents
