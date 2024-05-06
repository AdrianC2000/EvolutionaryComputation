import numpy as np
from numpy import ndarray


class FitnessFunction:

    def __init__(self, fitness_function: str):
        self.__FUNCTIONS = {
            "square_sum": self._square_sum,
            "katsuura": self._katsuura,
            "rana": self._rana
        }

        self.selected_function = self.__FUNCTIONS[fitness_function]

    def _square_sum(self, individual: ndarray) -> float:
        return float(np.sum(np.power(individual, 2)))

    # katsuura
    def _katsuura(self, x: ndarray) -> float:
        sum_rana = 0

        for i in range(len(x) - 1):
            sqrt_term1 = np.sqrt(np.abs(x[i + 1] + x[i] + 1))
            sqrt_term2 = np.sqrt(np.abs(x[i + 1] - x[i] + 1))

            term1 = x[i] * np.cos(sqrt_term1) * np.sin(sqrt_term2)
            term2 = (1 + x[i + 1]) * np.sin(sqrt_term1) * np.cos(sqrt_term2)

            sum_rana += term1 + term2

        return sum_rana

    # rana
    def _rana(self, x) -> float:
        D = len(x)
        term = 1.0 * 10 / D ** 2
        for i in range(D):
            inner_sum = 0.0
            for j in range(1, 33):
                term_j = 2 ** j * x[i]
                inner_sum += np.abs(term_j - np.round(term_j)) / (2 ** j)
            term *= (1 + (i + 1) * inner_sum) ** (10.0 / D ** 1.2)
        return term - 10 / D ** 2