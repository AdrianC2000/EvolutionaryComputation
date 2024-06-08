from mealpy import FloatVar
from mealpy.bio_based import SOA
from mealpy.utils.agent import Agent

from seagull_optimization_algorithm.parameters.soa_parameters import SoaParameters


class SoaAlgorithm:

    def __init__(self, parameters: SoaParameters):
        self.__parameters = parameters

    def find_result(self) -> Agent:
        min_bound, max_bound = self.__parameters.common_parameters.bounds
        problem = {
            "obj_func": self.__parameters.common_parameters.fitness_function,
            "bounds": FloatVar(lb=(min_bound,) * self.__parameters.common_parameters.variables_number,
                               ub=(max_bound,) * self.__parameters.common_parameters.variables_number),
            "minmax": self.__parameters.common_parameters.minmax,
            "log_to": "log.txt",
        }
        model = SOA.OriginalSOA(epoch=self.__parameters.common_parameters.epochs,
                                pop_size=self.__parameters.common_parameters.population_size)
        return model.solve(problem)
