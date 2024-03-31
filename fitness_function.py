import numpy as np
from numpy import ndarray


class FitnessFunction:

    @staticmethod
    def fitness_function(individual: ndarray) -> float:
        return float(np.sum(np.power(individual, 3)))
