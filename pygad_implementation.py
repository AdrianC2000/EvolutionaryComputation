# na podstawie przykładu: https://pypi.org/project/pygad/1.0.18/
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pygad

from binary_encoder import BinaryEncoder
from crossover_algorithms.crossover_algorithms_pygad_binary import CrossoverAlgorithmsPyGADBinary
from crossover_algorithms.crossover_algorithms_pygad_real import CrossoverAlgorithmsPyGADReal
from fitness_functions import FitnessFunction
from mutation_algorithms.mutation_algorithms_pygad_real import MutationAlgorithmsPyGADReal

binary_precision = 6
num_generations = 200
sol_per_pop = 80
num_parents_mating = 50
crossover_probability = 0.8
mutation_percent_genes = 20
mutation_probability = 0.2

print("Please choose the chromosome representation:\n1. Binary\n2. Real")
# binary_choice = input()

# if binary_choice == '1':
is_binary = True
# elif binary_choice == '2':
#     is_binary = False
# else:
#     print("Invalid input. Please enter 1 or 2.")
#     sys.exit()

print("Please choose the fitness function:\n1. Square Sum\n2. Rana\n3. Hyperellipsoid\n4. Schwefel\n5. Ackley")
# fitness_function_choice = input()

# if fitness_function_choice == '1':
#     fitness_function = 'square_sum'
# elif fitness_function_choice == '2':
fitness_function = 'rana'
# elif fitness_function_choice == '3':
#     fitness_function = 'hyperellipsoid'
# elif fitness_function_choice == '4':
#     fitness_function = 'schwefel'
# elif fitness_function_choice == '5':
#     fitness_function = 'ackley'
# else:
#     print("Invalid input")
#     sys.exit()

# num_genes = int(input("Choose number of genes: "))
num_genes = 8

random_mutation_min_val, random_mutation_max_val = (0, 2) if is_binary else (-32.768, 32.768)
gene_type = int if is_binary else float
func = FitnessFunction(fitness_function)
func.selected_function(np.array((0, 2)))
init_range_low, init_range_high = (0, 2) if is_binary else (
    func.suggested_bounds[0][0], func.suggested_bounds[1][0])
encoder = BinaryEncoder(binary_precision, func.suggested_bounds[0][0], func.suggested_bounds[1][0])
binary_chain_length = encoder.get_binary_chain_length()
num_bits = binary_chain_length
if is_binary:
    num_genes *= num_bits

print("Please choose extreme:\n1. Min\n2. Max")
# min_max_choice = input()

# if min_max_choice == '1':
is_min = True
# elif min_max_choice == '2':
#     is_min = False
# else:
#     print("Invalid input")
#     sys.exit()

print("Please choose selection method:\n1. tournament\n2. rws\n3. random")
# parent_selection_type_choice = input()

# if parent_selection_type_choice == '1':
parent_selection_type = 'tournament'
# elif parent_selection_type_choice == '2':
#     parent_selection_type = 'rws'
# elif parent_selection_type_choice == '3':
#     parent_selection_type = 'random'
# else:
#     print("Invalid input")
#     sys.exit()

if is_binary:
    # print("Please choose crossover method:\n1. single_point\n2. two_points\n3. uniform\n4. three-point\n5. grain\n6. "
    #       "mssx\n7. three-parent\n8. nonuniform")
    # crossover_choice = input()
    crossovers_pygad_binary = CrossoverAlgorithmsPyGADBinary(encoder.get_binary_chain_length())
    # if crossover_choice == '1':
    crossover_type = 'single_point'
    # elif crossover_choice == '2':
    #     crossover_type = 'two_points'
    # elif crossover_choice == '3':
    #     crossover_type = 'uniform'
    # elif crossover_choice == '4':
    #     crossover_type = crossovers_pygad_binary.get_methods()['three-point']
    # elif crossover_choice == '5':
    #     crossover_type = crossovers_pygad_binary.get_methods()['grain']
    # elif crossover_choice == '6':
    #     crossover_type = crossovers_pygad_binary.get_methods()['mssx']
    # elif crossover_choice == '7':
    #     crossover_type = crossovers_pygad_binary.get_methods()['three-parent']
    # elif crossover_choice == '8':
    #     crossover_type = crossovers_pygad_binary.get_methods()['nonuniform']
    # else:
    #     print("Invalid input")
    #     sys.exit()
else:
    # print("Please choose crossover method:\n1. arithmetical\n2. linear\n3. blend_alpha\n4. blend_alpha_and_beta\n5. "
    #       "average\n6. unfair_average\n7. gaussian_uniform")
    # crossover_choice = input()
    crossovers_pygad_real = CrossoverAlgorithmsPyGADReal((init_range_low, init_range_high), is_min, fitness_function)
    # if crossover_choice == '1':
    crossover_type = crossovers_pygad_real.get_methods()['arithmetical']
    # elif crossover_choice == '2':
    #     crossover_type = crossovers_pygad_real.get_methods()['linear']
    # elif crossover_choice == '3':
    #     crossover_type = crossovers_pygad_real.get_methods()['blend_alpha']
    # elif crossover_choice == '4':
    #     crossover_type = crossovers_pygad_real.get_methods()['blend_alpha_and_beta']
    # elif crossover_choice == '5':
    #     crossover_type = crossovers_pygad_real.get_methods()['average']
    # elif crossover_choice == '6':
    #     crossover_type = crossovers_pygad_real.get_methods()['unfair_average']
    # elif crossover_choice == '7':
    #     crossover_type = crossovers_pygad_real.get_methods()['gaussian_uniform-parent']
    # else:
    #     print("Invalid input")
    #     sys.exit()

# print("Please choose mutation method:\n1. random\n2. swap")
# if not is_binary:
#     print("3. gaussian")
# mutation_choice = input()
# if mutation_choice == '1':
if not is_binary:
    mutations_pygad_real = MutationAlgorithmsPyGADReal(mutation_probability, (init_range_low, init_range_high))
    mutation_type = mutations_pygad_real.get_methods()["gaussian"]
else:
    mutation_type = 'random'
# elif mutation_choice == '2':
#     mutation_type = 'swap'
# elif mutation_choice == '3' and not is_binary:
#     mutations_pygad_real = MutationAlgorithmsPyGADReal(mutation_probability, (init_range_low, init_range_high))
#     mutation_type = mutations_pygad_real.get_methods()["gaussian"]
# else:
#     print("Invalid input")
#     sys.exit()


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
                       mutation_percent_genes=mutation_percent_genes,
                       mutation_probability=mutation_probability,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       crossover_probability=0.8,
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
