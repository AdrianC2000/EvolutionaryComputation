from genetic_algorithm import GeneticAlgorithm

if __name__ == "__main__":
    variables_number = 3
    bounds = (-10, 10)
    population_size = 100
    epochs_number = 100
    precision = 6
    selection_method = "tournament"  # TODO - params
    crossover_method = "single-point"
    crossover_probability = 0.5
    mutation_method = "single-point"
    mutation_probability = 0.5
    inversion_probability = 0.5
    elitism_probability = 0.5

    genetic_algorithm = GeneticAlgorithm(precision, bounds, selection_method)
    genetic_algorithm.find_best_solution(population_size, variables_number, epochs_number)
