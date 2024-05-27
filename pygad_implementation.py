# na podstawie przykładu: https://pypi.org/project/pygad/1.0.18/
import logging

import benchmark_functions as bf
import numpy as np
import pygad

from binary_encoder import BinaryEncoder

# Konfiguracja algorytmu genetycznego

is_binary = True
binary_precision = 6
num_bits = 6
num_genes = 3

func_ackley = bf.Ackley(n_dimensions=num_genes)
func_schwefel = bf.Schwefel(n_dimensions=num_genes)
func_rana = bf.Rana(n_dimensions=num_genes)

if is_binary:
    num_genes *= num_bits

func = func_schwefel

init_range_low, init_range_high = (0, 2) if is_binary else (
    func.suggested_bounds()[0][0], func.suggested_bounds()[1][0])
encoder = BinaryEncoder(binary_precision, func.suggested_bounds()[0][0], func.suggested_bounds()[1][0])


def fitness_func(ga_instance, solution, solution_idx, is_min=False):
    if is_binary:
        bit_str_combined = ''.join(solution.astype(str))
        individuals = [bit_str_combined[i * num_bits:(i + 1) * num_bits] for i in range(int(num_genes / num_bits))]
        solution = encoder.decode_individual(np.array(individuals))
    fitness = func(solution)
    return 1. / fitness if is_min else fitness


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

    ga_instance.logger.info("Min    = {min}".format(min=np.min(tmp)))
    ga_instance.logger.info("Max    = {max}".format(max=np.max(tmp)))
    ga_instance.logger.info("Average    = {average}".format(average=np.average(tmp)))
    ga_instance.logger.info("Std    = {std}".format(std=np.std(tmp)))
    ga_instance.logger.info("\r\n")


fitness_function = fitness_func
is_min = True
num_generations = 100
sol_per_pop = 80
num_parents_mating = 50
init_range_low, init_range_high = (0, 2) if is_binary else (
    func.suggested_bounds()[0][0], func.suggested_bounds()[1][0])
random_mutation_min_val, random_mutation_max_val = (0, 2) if is_binary else (-32.768, 32.768)
mutation_num_genes = 1
parent_selection_type = "tournament"
crossover_type = "single_point"
mutation_type = "random"
gene_type = int if is_binary else float
# Konfiguracja logowania

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
                       mutation_num_genes=mutation_num_genes,
                       mutation_by_replacement=is_binary,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
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

# sztuczka: odwracamy my narysował nam się oczekiwany wykres dla problemu minimalizacji
ga_instance.best_solutions_fitness = [1. / x if is_min else x for x in ga_instance.best_solutions_fitness]
ga_instance.plot_fitness()
