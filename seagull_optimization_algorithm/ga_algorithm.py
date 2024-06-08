import logging

import numpy as np
from mealpy.utils.agent import Agent
from pygad import pygad

from binary_encoder import BinaryEncoder
from seagull_optimization_algorithm.parameters.ga_parameters import GaParameters

level = logging.DEBUG
name = 'logfile.txt'
logger = logging.getLogger(name)
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

BINARY_PRECISION = 8


def fitness_func(ga_instance, solution, solution_idx, parameters, is_min=False):
    encoder = BinaryEncoder(BINARY_PRECISION, parameters.common_parameters.bounds[0],
                            parameters.common_parameters.bounds[1])
    binary_chain_length = encoder.get_binary_chain_length()
    num_bits = binary_chain_length
    num_genes = parameters.common_parameters.variables_number

    if parameters.is_binary:
        num_genes *= num_bits
        bit_str_combined = ''.join(solution.astype(str))
        individuals = [bit_str_combined[i * num_bits:(i + 1) * num_bits] for i in range(int(num_genes / num_bits))]
        solution = encoder.decode_individual(np.array(individuals))
    fitness = parameters.common_parameters.fitness_function(solution)
    return 1. / fitness if is_min else fitness


def on_generation(ga_instance):
    ga_instance.logger.info(f"Epoch no {ga_instance.generations_completed} done, fitness: {ga_instance.best_solution()[1]}")


class GaAlgorithm:

    def __init__(self, parameters: GaParameters):
        self.__parameters = parameters

    def find_result(self) -> Agent:
        init_range_low, init_range_high = (0, 2) if self.__parameters.is_binary else (
            self.__parameters.common_parameters.bounds[0], self.__parameters.common_parameters.bounds[1])
        is_min = self.__parameters.common_parameters.minmax == "min"
        gene_type = int if self.__parameters.is_binary else float
        random_mutation_min_val, random_mutation_max_val = (0, 2) if self.__parameters.is_binary else (
        self.__parameters.common_parameters.bounds[0], self.__parameters.common_parameters.bounds[1])

        encoder = BinaryEncoder(BINARY_PRECISION, self.__parameters.common_parameters.bounds[0],
                                self.__parameters.common_parameters.bounds[1])
        binary_chain_length = encoder.get_binary_chain_length()
        num_bits = binary_chain_length
        num_genes = self.__parameters.common_parameters.variables_number
        if self.__parameters.is_binary:
            num_genes *= num_bits
        ga_instance = pygad.GA(num_generations=self.__parameters.common_parameters.epochs,
                               sol_per_pop=self.__parameters.common_parameters.population_size,
                               num_parents_mating=int(0.6 * self.__parameters.common_parameters.population_size),
                               num_genes=num_genes,
                               fitness_func=lambda ga_instance, solution, solution_idx:
                               fitness_func(ga_instance, solution, solution_idx, is_min=is_min,
                                            parameters=self.__parameters),
                               init_range_low=init_range_low,
                               init_range_high=init_range_high,
                               gene_type=gene_type,
                               mutation_by_replacement=True,
                               mutation_percent_genes=10,
                               mutation_probability=0.2,
                               parent_selection_type="sss",
                               keep_parents=2,
                               crossover_type="uniform",
                               crossover_probability=0.8,
                               mutation_type="random",
                               keep_elitism=1,
                               K_tournament=3,
                               random_mutation_max_val=random_mutation_max_val,
                               random_mutation_min_val=random_mutation_min_val,
                               logger=logger,
                               on_generation=lambda ga_instance: on_generation(ga_instance),
                               parallel_processing=['thread', 4])

        ga_instance.run()
        ga_instance.best_solutions_fitness = [1. / x if is_min else x for x in ga_instance.best_solutions_fitness]
        if self.__parameters.is_binary:
            best_solution_binary = ga_instance.best_solution()[0]
            bit_str_combined = ''.join(best_solution_binary.astype(str))
            individuals = [bit_str_combined[i * num_bits:(i + 1) * num_bits] for i in range(int(num_genes / num_bits))]
            individuals_ndarray = [np.array(x) for x in individuals]
            best_solution = [float(encoder.decode_individual(x))
                             for x in individuals_ndarray]
            return best_solution, ga_instance.best_solution()[1]
        return ga_instance.best_solution()[0], ga_instance.best_solution()[1]
