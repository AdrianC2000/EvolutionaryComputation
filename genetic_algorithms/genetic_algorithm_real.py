from typing import Tuple

import numpy as np
from numpy import ndarray

from crossover_algorithms.crossover_algorithms_real import CrossoverAlgorithmsReal
from genetic_algorithms.genetic_algorithm import GeneticAlgorithm
from mutation_algorithms.mutation_algorithms_real import MutationAlgorithmsReal


class GeneticAlgorithmReal(GeneticAlgorithm):

    def __init__(self, bounds: Tuple[int, int], variables_number: int, selection_method: str,
                 crossover_method: str, crossover_probability: float, mutation_method: str, mutation_rate: float,
                 elitism_ratio: float, fitness_function: str, is_min_searched: bool = False, tournaments_count: int = 3,
                 fraction_selected: float = 0.34):
        super().__init__(bounds, variables_number, selection_method, elitism_ratio, is_min_searched, tournaments_count,
                         fraction_selected, fitness_function)
        self.__crossover_algorithms = CrossoverAlgorithmsReal(crossover_method, crossover_probability, bounds)
        self.__mutation_algorithms = MutationAlgorithmsReal(mutation_method, mutation_rate, variables_number, bounds)

    def find_best_solution(self, population_size: int, epochs_number: int) -> Tuple[ndarray, float]:
        population = self._initialize_population(population_size, self._variables_number)
        best_fitness, best_individual = 0, None

        with open('../ga_results.txt', 'w') as file:
            for epoch in range(epochs_number):
                new_best_individual, new_best_fitness = self._get_best_individual(population)
                if new_best_fitness > best_fitness:
                    best_individual = new_best_individual
                    best_fitness = new_best_fitness

                selected_parents_population = self._selection_algorithms.select_parents(
                    population,
                    tournaments_count=self._tournaments_count,
                    fraction_selected=self._fraction_selected,
                    is_min_searched=self._is_min_searched
                )

                children = self.__crossover_algorithms.perform_crossover(selected_parents_population)
                children_mutated = self.__mutation_algorithms.perform_mutation(children)

                population = self._replace_population(population, children_mutated)
                self._fitness_history.append(new_best_fitness)
                fitness_values = [self._fitness_function(individual) for individual in population]
                self._average_fitness_history.append(np.mean(fitness_values))
                self._std_dev_fitness_history.append(np.std(fitness_values))
                file.write(f'Epoch {epoch + 1}, Best Fitness: {best_fitness}\n')

        return best_individual, best_fitness
