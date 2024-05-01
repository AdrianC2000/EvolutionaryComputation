from typing import Tuple

import numpy as np
from numpy import ndarray

from mutation_algorithms.mutation_algorithms import MutationAlgorithms


class MutationAlgorithmsReal(MutationAlgorithms):

    def __init__(self, mutation_method: str, mutation_rate: float, variables_number: int, bounds: Tuple[int, int]):
        __METHODS = {
            "uniform": self._uniform_mutation,
            "gaussian": self._gaussian_mutation
        }
        super().__init__(mutation_method, mutation_rate, variables_number, __METHODS)
        self.__bounds = bounds

    def _uniform_mutation(self, population: ndarray) -> ndarray:
        mutated_population = np.copy(population)

        for chromosome_index in range(len(population)):
            if np.random.rand() < self._mutation_rate:
                gene_to_mutate_index = np.random.randint(0, self._variables_number)
                new_gene_value = np.random.uniform(self.__bounds[0], self.__bounds[1])
                mutated_population[chromosome_index] = self._switch_gene(population[chromosome_index],
                                                                         gene_to_mutate_index, new_gene_value)
        return mutated_population

    @staticmethod
    def _switch_gene(old_chromosome: ndarray, gene_to_mutate_index: int, new_gene_value: float) -> ndarray:
        old_chromosome[gene_to_mutate_index] = new_gene_value
        return old_chromosome

    def _gaussian_mutation(self, population: ndarray) -> ndarray:
        mutated_population = np.copy(population)

        for chromosome_index in range(len(population)):
            if np.random.rand() < self._mutation_rate:
                random_normal = np.random.normal()
                mutated_population[chromosome_index] = self._mutate_chromosome(population[chromosome_index],
                                                                               random_normal)
        return mutated_population

    def _mutate_chromosome(self, chromosome: ndarray, random_normal: float) -> ndarray:
        new_chromosome = np.copy(chromosome)
        for index, gene in enumerate(chromosome):
            if index % 2 == 0:
                new_gene = gene + random_normal
                if self.__bounds[0] <= new_gene <= self.__bounds[1]:
                    new_chromosome[index] = new_gene
            else:
                new_gene = gene - random_normal
                if self.__bounds[0] <= new_gene <= self.__bounds[1]:
                    new_chromosome[index] = new_gene
        return new_chromosome
