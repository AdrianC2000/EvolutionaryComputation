from typing import Tuple

import numpy as np
from numpy import ndarray

from binary_encoder import BinaryEncoder
from crossover_algorithms.crossover_algorithms_binary import CrossoverAlgorithmsBinary
from genetic_algorithms.genetic_algorithm import GeneticAlgorithm
from mutation_algorithms.mutation_algorithms_binary import MutationAlgorithmsBinary


class GeneticAlgorithmBinary(GeneticAlgorithm):

    def __init__(self, precision: int, bounds: Tuple[int, int], variables_number: int, selection_method: str,
                 crossover_method: str, crossover_probability: float, mutation_method: str, mutation_rate: float,
                 inversion_probability: float, elitism_ratio: float, fitness_function: str,
                 is_min_searched: bool = False, tournaments_count: int = 3, fraction_selected: float = 0.34):
        super().__init__(bounds, variables_number, selection_method, elitism_ratio, is_min_searched, tournaments_count,
                         fraction_selected, fitness_function)
        self.__binary_encoder = BinaryEncoder(precision, bounds[0], bounds[1])
        binary_chain_length = self.__binary_encoder.get_binary_chain_length()

        self.__crossover_algorithms = CrossoverAlgorithmsBinary(crossover_method, binary_chain_length,
                                                                crossover_probability)
        self.__mutation_algorithms = MutationAlgorithmsBinary(mutation_method, mutation_rate, variables_number,
                                                              binary_chain_length)
        self.__inversion_probability = inversion_probability

    def find_best_solution(self, population_size: int, epochs_number: int) -> Tuple[ndarray, float]:
        population = self._initialize_population(population_size, self._variables_number)
        best_fitness = 10000 if self._is_min_searched else -10000
        best_individual = None

        with open('../ga_results.txt', 'w') as file:
            for epoch in range(epochs_number):
                new_best_individual, new_best_fitness = self._get_best_individual(population)
                if (not self._is_min_searched and new_best_fitness > best_fitness) or (
                        self._is_min_searched and new_best_fitness < best_fitness):
                    best_individual = new_best_individual
                    best_fitness = new_best_fitness

                selected_parents_population = self._selection_algorithms.select_parents(
                    population,
                    tournaments_count=self._tournaments_count,
                    fraction_selected=self._fraction_selected,
                    is_min_searched=self._is_min_searched
                )

                selected_parents_encoded = self.__binary_encoder.encode_population(selected_parents_population)
                children_encoded = self.__crossover_algorithms.perform_crossover(selected_parents_encoded)
                children_mutated = self.__mutation_algorithms.perform_mutation(children_encoded)

                children = self.__binary_encoder.decode_population(children_mutated)
                children_inverted = self._invert_segments(children)

                population = self._replace_population(population, children_inverted)
                self.fitness_history.append(new_best_fitness)
                fitness_values = [self._fitness_function(individual) for individual in population]
                self.average_fitness_history.append(np.mean(fitness_values))
                self.std_dev_fitness_history.append(np.std(fitness_values))
                file.write(f'Epoch {epoch + 1}, Best Fitness: {best_fitness}\n')

        return best_individual, best_fitness

    def _invert_segments(self, children: ndarray) -> ndarray:
        length = self.__binary_encoder.get_binary_chain_length()
        for individual in children:
            if np.random.random() < self.__inversion_probability:
                point1, point2 = sorted(np.random.choice(range(1, length), 2, replace=False))
                individual[point1:point2] = individual[point1:point2][::-1]
        return children
