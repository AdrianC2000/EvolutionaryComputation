from typing import Tuple, List

import numpy as np
from numpy import ndarray

from binary_encoder import BinaryEncoder
from crossover_algorithms import CrossoverAlgorithms
from fitness_function import FitnessFunction
from mutation_algorithms import MutationAlgorithms
from selection_algorithms import SelectionAlgorithms


class GeneticAlgorithm:

    def __init__(self, precision: int, bounds: Tuple[int, int], variables_number: int, selection_method: str,
                 crossover_method: str, crossover_probability: float, mutation_method: str, mutation_rate: float,
                 inversion_probability : float, elitism_ratio: float, is_min_searched: bool = False,
                 tournaments_count: int = 3, fraction_selected: float = 0.34):
        self.__bounds = bounds
        self.__binary_encoder = BinaryEncoder(precision, bounds[0], bounds[1])
        self.__variables_number = variables_number

        self.fitness_history = []  # Store fitness of the best individual each iteration
        self.average_fitness_history = []  # Store average fitness each iteration
        self.std_dev_fitness_history = [] 

        binary_chain_length = self.__binary_encoder.get_binary_chain_length()
        self.__selection_algorithms = SelectionAlgorithms(selection_method)
        self.__tournaments_count = tournaments_count
        self.__fraction_selected = fraction_selected
        self.__crossover_algorithms = CrossoverAlgorithms(crossover_method, binary_chain_length, crossover_probability)
        self.__mutation_algorithms = MutationAlgorithms(mutation_method, mutation_rate,
                                                        self.__variables_number, binary_chain_length)
        self.__elitism_ratio = elitism_ratio
        self.__inversion_probability = inversion_probability
        self.__is_min_searched = is_min_searched

    def find_best_solution(self, population_size: int, epochs_number: int) -> Tuple[ndarray, float]:
        population = self._initialize_population(population_size, self.__variables_number)
        best_fitness, best_individual = 0, None

        with open('ga_results.txt', 'w') as file:
            for epoch in range(epochs_number):
                new_best_individual, new_best_fitness = self._get_best_individual(population)
                if new_best_fitness > best_fitness:
                    best_individual = new_best_individual
                    best_fitness = new_best_fitness

                selected_parents_population = self.__selection_algorithms.select_parents(
                    population,
                    tournaments_count=self.__tournaments_count,
                    fraction_selected=self.__fraction_selected,
                    is_min_searched=self.__is_min_searched
                )

                selected_parents_encoded = self.__binary_encoder.encode_population(selected_parents_population)
                children_encoded = self.__crossover_algorithms.perform_crossover(selected_parents_encoded)
                children_mutated = self.__mutation_algorithms.perform_mutation(children_encoded)
                children = self.__binary_encoder.decode_population(children_mutated)
                children = self._invert_segments(children)
                population = self._replace_population(population, children)
                self.fitness_history.append(new_best_fitness)
                fitness_values = [FitnessFunction.fitness_function(individual) for individual in population]
                self.average_fitness_history.append(np.mean(fitness_values))
                self.std_dev_fitness_history.append(np.std(fitness_values))
                file.write(f'Epoch {epoch + 1}, Best Fitness: {best_fitness}\n')

        return best_individual, best_fitness

    def _initialize_population(self, population_size: int, variables_number: int) -> ndarray:
        return np.random.uniform(self.__bounds[0], self.__bounds[1], size=(population_size, variables_number))

    @staticmethod
    def _get_best_individual(population: ndarray) -> Tuple[ndarray, float]:
        fitnesses = [FitnessFunction.fitness_function(individual) for individual in population]
        best_index = np.argmax(fitnesses)
        return population[best_index], fitnesses[best_index]

    def _invert_segments(self, children: ndarray) -> ndarray:
        length = self.__binary_encoder.get_binary_chain_length()
        for individual in children:
            if np.random.random() < self.__inversion_probability:
                point1, point2 = sorted(np.random.choice(range(1, length), 2, replace=False))
                individual[point1:point2] = individual[point1:point2][::-1]
        return children

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
        if self.__is_min_searched:
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
