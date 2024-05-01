from numpy import ndarray


class MutationAlgorithms:
    def __init__(self, mutation_method: str, mutation_rate: float, variables_number: int, methods: dict):
        self.__selected_method = methods[mutation_method]
        self._mutation_method = mutation_method
        self._mutation_rate = mutation_rate
        self._variables_number = variables_number

    def perform_mutation(self, population: ndarray) -> ndarray:
        return self.__selected_method(population)
