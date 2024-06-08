from fitness_functions import FitnessFunction
from seagull_optimization_algorithm.ga_algorithm import GaAlgorithm
from seagull_optimization_algorithm.parameters.common_parameters import CommonParameters
from seagull_optimization_algorithm.parameters.ga_parameters import GaParameters
from seagull_optimization_algorithm.parameters.soa_parameters import SoaParameters

from seagull_optimization_algorithm.soa_algorithm import SoaAlgorithm

FITNESS_FUNCTION = FitnessFunction("square_sum")

COMMON_PARAMETERS = CommonParameters(
    epochs=1000,
    population_size=100,
    fitness_function=FITNESS_FUNCTION.selected_function,
    bounds=(FITNESS_FUNCTION.suggested_bounds[0], FITNESS_FUNCTION.suggested_bounds[1]),
    variables_number=5,
    minmax="max"
)

GA_PARAMETERS = GaParameters(
    common_parameters=COMMON_PARAMETERS,
    is_binary=True
)


def compare_algorithms():
    soa_algorithm = SoaAlgorithm(SoaParameters(COMMON_PARAMETERS))
    seagull_optimization_result = soa_algorithm.find_result()
    print(f"Best solution for seagull optimization: {seagull_optimization_result.solution}")
    print(f"Best fitness: {seagull_optimization_result.target.fitness}")
    ga_algorithm = GaAlgorithm(GA_PARAMETERS)
    best_solution, best_fitness = ga_algorithm.find_result()
    print(f"Best solution for genetic algorithm: {best_solution}")
    print(f"Best fitness: {best_fitness}")


if __name__ == "__main__":
    compare_algorithms()
