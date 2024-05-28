# na podstawie przykładu: https://pypi.org/project/pygad/1.0.18/
import logging

import benchmark_functions as bf
import numpy as np
import pygad
import matplotlib.pyplot as plt

from binary_encoder import BinaryEncoder
from fitness_functions import FitnessFunction
from crossover_algorithms.crossover_algorithms_pygad_binary import CrossoverAlgorithmsPyGADBinary
from crossover_algorithms.crossover_algorithms_pygad_real import CrossoverAlgorithmsPyGADReal
from mutation_algorithms.mutation_algorithms_pygad_real import MutationAlgorithmsPyGADReal

is_binary = False
binary_precision = 6
num_genes = 5

func_ackley = bf.Ackley(n_dimensions=num_genes)
func_schwefel = bf.Schwefel(n_dimensions=num_genes)
func_rana = bf.Rana(n_dimensions=num_genes)

fitness_function = "schwefel"

func = FitnessFunction(fitness_function)
func.selected_function(np.array((0, 2)))

init_range_low, init_range_high = (0, 2) if is_binary else (
    func.suggested_bounds[0][0], func.suggested_bounds[1][0])
encoder = BinaryEncoder(binary_precision, func.suggested_bounds[0][0], func.suggested_bounds[1][0])
binary_chain_length = encoder.get_binary_chain_length()
num_bits = binary_chain_length
if is_binary:
    num_genes *= num_bits


def fitness_func(ga_instance, solution, solution_idx, is_min=False):
    if is_binary:
        bit_str_combined = ''.join(solution.astype(str))
        individuals = [bit_str_combined[i * num_bits:(i + 1) * num_bits] for i in range(int(num_genes / num_bits))]
        solution = encoder.decode_individual(np.array(individuals))
    fitness = func.selected_function(solution)
    return 1. / fitness if is_min else fitness


stds = []
averages = []


def on_generation(ga_instance, is_min=False):
    ga_instance.logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)
    fitness = 1. / solution_fitness if is_min else solution_fitness
    ga_instance.logger.info("Best    = {fitness}".format(fitness=fitness))
    ga_instance.logger.info("Individual    = {solution}".format(solution=repr(solution)))
    if is_binary:
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
    stds.append(std)
    averages.append(average)
    ga_instance.logger.info("\r\n")


is_min = True
num_generations = 300
sol_per_pop = 100
num_parents_mating = 100
random_mutation_min_val, random_mutation_max_val = (0, 2) if is_binary else (-32.768, 32.768)
parent_selection_type = "tournament"

crossovers_pygad_binary = CrossoverAlgorithmsPyGADBinary(encoder.get_binary_chain_length())
crossovers_pygad_real = CrossoverAlgorithmsPyGADReal((init_range_low, init_range_high), is_min, fitness_function)
crossovers_method = crossovers_pygad_real.get_methods()["unfair_average"]
crossover_type = "single_point"
mutation_type = "random"
mutation_probability = 0.3
mutations_pygad_real = MutationAlgorithmsPyGADReal(mutation_probability, (init_range_low, init_range_high))
mutation_method = mutations_pygad_real.get_methods()["gaussian"]
gene_type = int if is_binary else float

level = logging.DEBUG
name = 'logfile.txt'
logger = logging.getLogger(name)
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# Właściwy algorytm genetyczny

ga_instance = pygad.GA(num_generations=num_generations,
                       sol_per_pop=sol_per_pop,
                       num_parents_mating=num_parents_mating,
                       num_genes=num_genes,
                       fitness_func=lambda ga_instance, solution, solution_idx: fitness_func(ga_instance, solution,
                                                                                             solution_idx,
                                                                                             is_min=is_min),
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       gene_type=gene_type,
                       mutation_by_replacement=is_binary,
                       mutation_percent_genes=20,
                       mutation_probability=mutation_probability,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       crossover_probability=0.8,
                       mutation_type=mutation_method,
                       keep_elitism=1,
                       K_tournament=3,
                       random_mutation_max_val=random_mutation_max_val,
                       random_mutation_min_val=random_mutation_min_val,
                       logger=logger,
                       on_generation=lambda ga_instance: on_generation(ga_instance, is_min=is_min),
                       parallel_processing=['thread', 4])

ga_instance.run()

best = ga_instance.best_solution()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
solution_fitness = 1. / solution_fitness if is_min else solution_fitness
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

ga_instance.best_solutions_fitness = [1. / x if is_min else x for x in ga_instance.best_solutions_fitness]
ga_instance.plot_fitness()

generations = list(range(len(stds)))

plt.plot(generations, stds, linestyle='-', color='b')
plt.title('Generation vs. Standard Deviation')
plt.xlabel('Generation')
plt.ylabel('Standard Deviation')
plt.show()

plt.plot(generations, averages, linestyle='-', color='b')
plt.title('Generation vs. Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.show()
