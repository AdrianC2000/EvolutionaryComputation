from genetic_algorithm import GeneticAlgorithm

if __name__ == "__main__":
    variables_number = 3
    bounds = (-10, 10)
    population_size = 100
    epochs_number = 200
    precision = 6
    selection_method = "best"
    tournaments_count = 3
    fraction_selected = 0.34
    crossover_method = "mssx"
    crossover_probability = 0.5
    mutation_method = "single-point"
    mutation_rate = 0.1
    inversion_probability = 0.1
    elitism_ratio = 0.5
    is_min_searched = False

    genetic_algorithm = GeneticAlgorithm(precision, bounds, variables_number, selection_method, crossover_method,
                                         crossover_probability, mutation_method, mutation_rate, inversion_probability,
                                         elitism_ratio, is_min_searched, tournaments_count, fraction_selected)
    best_individual, best_fitness = genetic_algorithm.find_best_solution(population_size, epochs_number)
    print(f"Best found individual: {best_individual}, with fitness: {best_fitness}")
