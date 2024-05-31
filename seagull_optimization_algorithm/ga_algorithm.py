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

BINARY_PRECISION = 6


def fitness_func(ga_instance, solution, solution_idx, parameters, is_min=False):
    func = parameters.common_parameters.fitness_function
    encoder = BinaryEncoder(BINARY_PRECISION, func.suggested_bounds[0][0], func.suggested_bounds[1][0])
    binary_chain_length = encoder.get_binary_chain_length()
    num_bits = binary_chain_length
    num_genes = parameters.common_parameters.variables_number

    if parameters.is_binary:
        bit_str_combined = ''.join(solution.astype(str))
        individuals = [bit_str_combined[i * num_bits:(i + 1) * num_bits] for i in range(int(num_genes / num_bits))]
        solution = encoder.decode_individual(np.array(individuals))
    fitness = func.selected_function(solution)
    return 1. / fitness if is_min else fitness


def on_generation(ga_instance, parameters, is_min=False):
    func = parameters.common_parameters.fitness_function
    encoder = BinaryEncoder(BINARY_PRECISION, func.suggested_bounds[0][0], func.suggested_bounds[1][0])
    binary_chain_length = encoder.get_binary_chain_length()
    num_bits = binary_chain_length
    num_genes = parameters.common_parameters.variables_number

    ga_instance.logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)
    fitness = 1. / solution_fitness if is_min else solution_fitness
    ga_instance.logger.info("Best    = {fitness}".format(fitness=fitness))
    ga_instance.logger.info("Individual    = {solution}".format(solution=repr(solution)))
    if parameters.is_binary:
        bit_str_combined = ''.join(solution.astype(str))
        individuals = [bit_str_combined[i * num_bits:(i + 1) * num_bits] for i in range(int(num_genes / num_bits))]
        solution_real = encoder.decode_individual(np.array(individuals))
        ga_instance.logger.info("Individual real    = {solution}".format(solution=repr(solution_real)))

    tmp = [1. / x if is_min else x for x in
           ga_instance.last_generation_fitness]  # ponownie odwrotność by zrobić sobie dobre statystyki
    std = np.std(tmp)
    average = np.average(tmp)
    ga_instance.logger.info("Min    = {min}".format(min=np.min(tmp)))
    ga_instance.logger.info("Max    = {max}".format(max=np.max(tmp)))
    ga_instance.logger.info("Average    = {average}".format(average=average))
    ga_instance.logger.info("Std    = {std}".format(std=std))
    ga_instance.logger.info("\r\n")


class GaAlgorithm:

    def __init__(self, parameters: GaParameters):
        self.__parameters = parameters

    def find_result(self) -> Agent:
        is_min = self.__parameters.common_parameters.minmax == "min"
        gene_type = int if self.__parameters.is_binary else float
        random_mutation_min_val, random_mutation_max_val = (0, 2) if self.__parameters.is_binary else (-32.768, 32.768)
        ga_instance = pygad.GA(num_generations=self.__parameters.common_parameters.epochs,
                               sol_per_pop=self.__parameters.common_parameters.population_size,
                               num_parents_mating=int(0.6 * self.__parameters.common_parameters.population_size),
                               num_genes=self.__parameters.common_parameters.variables_number,
                               fitness_func=lambda ga_instance, solution, solution_idx:
                               fitness_func(ga_instance, solution, solution_idx, is_min=is_min,
                                            parameters=self.__parameters),
                               init_range_low=self.__parameters.common_parameters.bounds[0],
                               init_range_high=self.__parameters.common_parameters.bounds[1],
                               gene_type=gene_type,
                               mutation_by_replacement=self.__parameters.is_binary,
                               mutation_percent_genes=20,
                               mutation_probability=0.2,
                               parent_selection_type="tournament",
                               crossover_type="single_point",
                               crossover_probability=0.8,
                               mutation_type="random",
                               keep_elitism=1,
                               K_tournament=3,
                               random_mutation_max_val=random_mutation_max_val,
                               random_mutation_min_val=random_mutation_min_val,
                               logger=logger,
                               on_generation=lambda ga_instance: on_generation(ga_instance, is_min=is_min,
                                                                               parameters=self.__parameters),
                               parallel_processing=['thread', 4])

        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        solution_fitness = 1. / solution_fitness if is_min else solution_fitness
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        ga_instance.best_solutions_fitness = [1. / x if is_min else x for x in ga_instance.best_solutions_fitness]
        return ga_instance.best_solution()
