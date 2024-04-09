import numpy as np
from numpy import ndarray


class MutationAlgorithms:

    def __init__(self, mutation_method: str, mutation_rate: float, variables_number: int, binary_chain_length: int):
        self.__METHODS = {
            "single-point": self._single_point_mutation
        }
        self.__selected_method = self.__METHODS[mutation_method]
        self.__mutation_rate = mutation_rate
        self.__variables_number = variables_number
        self.__binary_chain_length = binary_chain_length

    def perform_mutation(self, population: ndarray) -> ndarray:
        return self.__selected_method(population)

    def _single_point_mutation(self, population: ndarray) -> ndarray:
        mutated_population = np.copy(population)

        for i in range(len(population)):
            if np.random.rand() < self.__mutation_rate:
                variable_to_mutate_index = np.random.randint(0, self.__variables_number)
                bit_to_mutate = np.random.randint(0, self.__binary_chain_length)
                variable_to_mutate = population[i, variable_to_mutate_index]
                mutated_population[i, variable_to_mutate_index] = self._switch_bit(variable_to_mutate, bit_to_mutate)
        return mutated_population
    
    # def _boundary_mutation(self, population: np.ndarray) -> np.ndarray:
    #     mutated_population = np.copy(population)

    #     for i in range(len(population)):
    #         if np.random.rand() < self.__mutation_rate:
    #             variable_to_mutate_index = np.random.randint(0, self.__variables_number)

    #             if np.random.rand() < 0.5:
    #                 boundary_value = self.__min_values[variable_to_mutate_index]
    #             else:
    #                 boundary_value = self.__max_values[variable_to_mutate_index]

    #             mutated_population[i, variable_to_mutate_index] = boundary_value
        
    #     return mutated_population

    @staticmethod
    def _switch_bit(chromosome: str, bit_to_mutate: int) -> str:
        return chromosome[:bit_to_mutate] \
            + ('0' if chromosome[bit_to_mutate] == '1' else '1') \
            + chromosome[bit_to_mutate + 1:]
