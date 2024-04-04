from genetic_algorithm import GeneticAlgorithm

if __name__ == "__main__":
    variables_number = 3
    bounds = (-10, 10)
    population_size = 100
    epochs_number = 100
    precision = 6
    selection_method = "roulette"  # TODO - params
    crossover_method = "single-point"
    crossover_probability = 0.5
    mutation_method = "single-point"
    mutation_rate = 0.1
    inversion_probability = 0.5
    elitism_probability = 0.5

    genetic_algorithm = GeneticAlgorithm(precision, bounds, variables_number, selection_method, crossover_method,
                                         mutation_method, mutation_rate)
    best_individual, best_fitness = genetic_algorithm.find_best_solution(population_size, epochs_number)
    print(f"Best found individual: {best_individual}, with fitness: {best_fitness}")
