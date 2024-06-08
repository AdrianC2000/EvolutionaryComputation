import numpy as np
from numpy import ndarray
import benchmark_functions as bf


class FitnessFunction:

    def __init__(self, fitness_function: str):
        self.__FUNCTIONS = {
            "square_sum": self._square_sum,
            "rana": self._rana,
            "hyperellipsoid": self._hyperellipsoid,
            "schwefel": self._schwefel,
            "ackley": self._ackley
        }
        self.__BOUNDS = {
            "square_sum": [-100, 100],
            "rana": bf.Rana().suggested_bounds(),
            "hyperellipsoid": bf.Hyperellipsoid().suggested_bounds(),
            "schwefel": bf.Schwefel().suggested_bounds(),
            "ackley": bf.Ackley().suggested_bounds()
        }

        self.suggested_bounds = self.__BOUNDS[fitness_function]
        self.selected_function = self.__FUNCTIONS[fitness_function]

    def _square_sum(self, individual: ndarray) -> float:
        self.suggested_bounds = [[-100], [100]]
        return float(np.sum(np.power(individual, 2)))

    def _rana(self, x: ndarray) -> float:
        func = bf.Rana(n_dimensions=x.size)
        self.suggested_bounds = func.suggested_bounds()
        return func(x.tolist())

    def _hyperellipsoid(self, x: ndarray) -> float:
        func = bf.Hyperellipsoid(n_dimensions=x.size)
        self.suggested_bounds = func.suggested_bounds()
        return func(x.tolist())

    def _schwefel(self, x: ndarray) -> float:
        func = bf.Schwefel(n_dimensions=x.size)
        self.suggested_bounds = func.suggested_bounds()
        return func(x.tolist())

    def _ackley(self, x: ndarray) -> float:
        func = bf.Ackley(n_dimensions=x.size)
        self.suggested_bounds = func.suggested_bounds()
        return func(x.tolist())
