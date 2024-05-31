from seagull_optimization_algorithm.ga_algorithm import GaAlgorithm
from seagull_optimization_algorithm.parameters.common_parameters import CommonParameters
from seagull_optimization_algorithm.parameters.ga_parameters import GaParameters
from seagull_optimization_algorithm.parameters.soa_parameters import SoaParameters

from seagull_optimization_algorithm.soa_algorithm import SoaAlgorithm

COMMON_PARAMETERS = CommonParameters(
    epochs=10,
    population_size=20,
    fitness_function_name="rana",
    bounds=(-10, 10),
    variables_number=20,
    minmax="max"
)

GA_PARAMETERS = GaParameters(
    common_parameters=COMMON_PARAMETERS,
    is_binary=True
)


def compare_algorithms():
    soa_algorithm = SoaAlgorithm(SoaParameters(COMMON_PARAMETERS))
    seagull_optimization_result = soa_algorithm.find_result()
    print(f"Best solution for seagull optimization: {seagull_optimization_result.solution}, "
          f"Best fitness: {seagull_optimization_result.target.fitness}")
    ga_algorithm = GaAlgorithm(GA_PARAMETERS)
    ga_algorithm_result = ga_algorithm.find_result()
    print(f"Best solution for genetic algorithm: {ga_algorithm_result.solution}, "
          f"Best fitness: {ga_algorithm_result.target.fitness}")


if __name__ == "__main__":
    compare_algorithms()
