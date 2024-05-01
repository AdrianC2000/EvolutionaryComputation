import numpy as np
from numpy import ndarray

from mutation_algorithms.mutation_algorithms import MutationAlgorithms


class MutationAlgorithmsBinary(MutationAlgorithms):

    def __init__(self, mutation_method: str, mutation_rate: float, variables_number: int, binary_chain_length: int):
        __METHODS = {
            "single-point": self._single_point_mutation
        }
        super().__init__(mutation_method, mutation_rate, variables_number, __METHODS)
        self.__binary_chain_length = binary_chain_length

    def _single_point_mutation(self, population: ndarray) -> ndarray:
        mutated_population = np.copy(population)

        for i in range(len(population)):
            if np.random.rand() < self._mutation_rate:
                variable_to_mutate_index = np.random.randint(0, self._variables_number)
                bit_to_mutate = np.random.randint(0, self.__binary_chain_length)
                variable_to_mutate = population[i, variable_to_mutate_index]
                mutated_population[i, variable_to_mutate_index] = self._switch_bit(variable_to_mutate, bit_to_mutate)
        return mutated_population

    @staticmethod
    def _switch_bit(chromosome: str, bit_to_mutate: int) -> str:
        return chromosome[:bit_to_mutate] \
            + ('0' if chromosome[bit_to_mutate] == '1' else '1') \
            + chromosome[bit_to_mutate + 1:]
