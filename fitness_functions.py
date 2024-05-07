import numpy as np
from numpy import ndarray
import benchmark_functions as bf


class FitnessFunction:

    def __init__(self, fitness_function: str):
        self.__FUNCTIONS = {
            "square_sum": self._square_sum,
            "rana": self._rana,
            "hyperellipsoid": self._hyperellipsoid
        }

        self.suggested_bounds = None
        self.selected_function = self.__FUNCTIONS[fitness_function]

    def _square_sum(self, individual: ndarray) -> float:
        return float(np.sum(np.power(individual, 2)))

    def _rana(self, x: ndarray) -> float:
        func = bf.Rana(n_dimensions=x.size)
        self.suggested_bounds = func.suggested_bounds()
        return func(x.tolist())

    def _hyperellipsoid(self, x: ndarray) -> float:
        func = bf.Hyperellipsoid(n_dimensions=x.size)
        self.suggested_bounds = func.suggested_bounds()
        return func(x.tolist())
