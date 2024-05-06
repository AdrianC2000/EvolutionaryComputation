from genetic_algorithms.genetic_algorithm_real import GeneticAlgorithmReal
from genetic_algorithms.genetic_algorithms_binary import GeneticAlgorithmBinary

if __name__ == "__main__":
    variables_number = 3
    bounds = (-10, 10)
    population_size = 100
    epochs_number = 200
    precision = 6
    selection_method = "best"
    tournaments_count = 3
    fraction_selected = 0.34
    crossover_probability = 0.5
    mutation_rate = 0.1
    inversion_probability = 0.1
    elitism_ratio = 0.5
    is_min_searched = False

    representation = "real"
    crossover_method = "gaussian_uniform"
    mutation_method = "gaussian"

    # representation = "binary"
    # crossover_method = "single-point"
    # mutation_method = "single-point"

    # square_sum, katsuura, rana
    fitness_function = "square_sum"

    if representation == "binary":
        genetic_algorithm = GeneticAlgorithmBinary(precision, bounds, variables_number, selection_method,
                                                   crossover_method, crossover_probability, mutation_method,
                                                   mutation_rate, inversion_probability, elitism_ratio,
                                                   fitness_function, is_min_searched, tournaments_count,
                                                   fraction_selected)
    else:
        genetic_algorithm = GeneticAlgorithmReal(bounds, variables_number, selection_method, crossover_method,
                                                 crossover_probability, mutation_method, mutation_rate, elitism_ratio,
                                                 fitness_function, is_min_searched, tournaments_count,
                                                 fraction_selected)

    best_individual, best_fitness = genetic_algorithm.find_best_solution(population_size, epochs_number)
    print(f"Best found individual: {best_individual}, with fitness: {best_fitness}")
