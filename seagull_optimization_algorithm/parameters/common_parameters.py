from typing import Tuple

from fitness_functions import FitnessFunction


class CommonParameters:

    def __init__(self, epochs: int, population_size: int, fitness_function: FitnessFunction,
                 bounds: Tuple[int, int], variables_number: int, minmax: str):
        self.epochs = epochs
        self.population_size = population_size
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.variables_number = variables_number
        self.minmax = minmax
