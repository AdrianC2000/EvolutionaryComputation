import random
from typing import List

import numpy as np


def create_subpopulations(population: List[List[int]]) -> List[List[List[int]]]:
    """ Methods creates a random number of subpopulation from the population
        Example output sizes for population of size 10: (5, 2, 3), (4, 2, 2, 2), (2, 2, 2, 2, 2) """
    subpopulations = []
    remaining_population = population.copy()

    while remaining_population:
        if len(remaining_population) in [2, 3]:
            subpopulations.append(remaining_population)
            break
        subpopulation_size = random.randint(2, len(remaining_population) // 2)
        subpopulation = random.sample(remaining_population, subpopulation_size)
        subpopulations.append(subpopulation)
        for item in subpopulation:
            remaining_population.remove(item)
    return subpopulations


def tournament_selection(population: List[List[int]]) -> List[int]:
    fitnesses = [fitness_function(individual) for individual in population]
    sample_number = 3 if len(population) > 2 else 2
    tournament_indexes = random.sample(range(len(population)), k=sample_number)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indexes]
    winner_index = tournament_indexes[np.argmax(tournament_fitnesses)]
    return population[winner_index]


def best_combinatorial_crossover(parent1: List[int], parent2: List[int]) -> List[List[int]]:
    all_possible_descendants = calculate_all_descendants(parent1, parent2)
    best_two_descendants = get_best_two_descendants(all_possible_descendants)
    return best_two_descendants


def calculate_all_descendants(parent1: List[int], parent2: List[int]) -> List[List[int]]:
    crossover_points = range(1, len(parent1))
    all_descendants = []
    for crossover_point in crossover_points:
        all_descendants.extend(single_point_crossover(parent1, parent2, crossover_point))
    return all_descendants


def single_point_crossover(parent_a: List[int], parent_b: List[int], crossover_point) -> List[List[int]]:
    descendant_1 = parent_a[:crossover_point] + parent_b[crossover_point:]
    descendant_2 = parent_b[:crossover_point] + parent_a[crossover_point:]
    return [descendant_1, descendant_2]


def get_best_two_descendants(all_descendants: List[List[int]]):
    return sorted(all_descendants, key=fitness_function, reverse=True)[:2]


def fitness_function(individual: List[int]) -> int:
    return sum(individual)


if __name__ == "__main__":
    population = [
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
        [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
    ]

    subpopulations = create_subpopulations(population)
    best_descendants_for_all_subpopulations = []
    for subpopulation in subpopulations:
        parent1 = tournament_selection(subpopulation)
        parent2 = tournament_selection(subpopulation)
        best_descendants_for_all_subpopulations.append(best_combinatorial_crossover(parent1, parent2))

    for best_descendant in best_descendants_for_all_subpopulations:
        print(best_descendant)
