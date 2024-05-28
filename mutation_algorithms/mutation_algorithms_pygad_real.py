from typing import Tuple

import numpy as np
from numpy import ndarray

from mutation_algorithms.mutation_algorithms import MutationAlgorithms


class MutationAlgorithmsPyGADReal:

    def __init__(self, mutation_rate: float, bounds: Tuple[int, int]):
        self.__METHODS = {
            "gaussian": self._gaussian_mutation
        }
        self.__mutation_rate = mutation_rate
        self.__bounds = bounds

    def get_methods(self):
        return self.__METHODS

    @staticmethod
    def _switch_gene(old_chromosome: ndarray, gene_to_mutate_index: int, new_gene_value: float) -> ndarray:
        old_chromosome[gene_to_mutate_index] = new_gene_value
        return old_chromosome

    def _gaussian_mutation(self, offspring, ga_instance) -> ndarray:
        mutated_population = np.copy(offspring)

        for chromosome_index in range(len(offspring)):
            if np.random.rand() < self.__mutation_rate:
                random_normal = np.random.normal()
                mutated_population[chromosome_index] = self._mutate_chromosome(offspring[chromosome_index],
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
