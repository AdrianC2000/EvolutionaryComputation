import numpy as np
from numpy import ndarray

from fitness_function import FitnessFunction


class SelectionAlgorithms:

    def __init__(self, selection_method: str):
        self.__METHODS = {
            "tournament": self._tournament_selection,
            "best": self._best_selection,
            "roulette": self._roulette_selection,
        }
        self.__selected_method = self.__METHODS[selection_method]

    def select_parents(self, population: ndarray, **kwargs) -> ndarray:
        return self.__selected_method(population, **kwargs)

    # TODO -> test is_min_searched
    @staticmethod
    def _tournament_selection(population: ndarray, **kwargs) -> ndarray:
        selected_parents = []
        tournaments_count = kwargs["tournaments_count"]
        is_min_searched = kwargs["is_min_searched"]

        for i in range(len(population)):

            if len(population) < tournaments_count:
                break

            tournament_indices = np.random.choice(len(population), size=tournaments_count, replace=False)
            tournament_individuals = population[tournament_indices]

            tournament_fitness = [FitnessFunction.fitness_function(individual) for individual in tournament_individuals]

            parent_index = np.argmin(tournament_fitness) if is_min_searched else np.argmax(tournament_fitness)

            selected_parents.append(tournament_individuals[parent_index])
            population = np.delete(population, tournament_indices, axis=0)

        return np.array(selected_parents)

    @staticmethod
    def _best_selection(population: ndarray, **kwargs) -> ndarray:
        fraction_selected = kwargs["fraction_selected"]
        is_min_searched = kwargs["is_min_searched"]
        fitness = [FitnessFunction.fitness_function(individual) for individual in population]
        selected_count = int(fraction_selected * len(fitness))

        if is_min_searched:
            extremes = np.partition(fitness, selected_count)[selected_count:]
        else:
            extremes = np.partition(fitness, -selected_count)[-selected_count:]

        indices = np.where(np.in1d(fitness, extremes))

        return population[indices]

    @staticmethod
    def _roulette_selection(population: ndarray, **kwargs) -> ndarray:
        fraction_selected = kwargs["fraction_selected"]
        is_min_searched = kwargs["is_min_searched"]

        fitness = [FitnessFunction.fitness_function(individual) for individual in population]
        selected_count = int(fraction_selected * len(fitness))

        if np.min(fitness) <= 0:
            fitness = fitness - np.min(fitness) + 1

        fitness_sum = np.sum(fitness)
        probabilities = fitness / fitness_sum

        if is_min_searched:
            probabilities = 1 / probabilities

        indices = np.random.choice(len(population), size=selected_count, p=probabilities)

        return population[indices]
