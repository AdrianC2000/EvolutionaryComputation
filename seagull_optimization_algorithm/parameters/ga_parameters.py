from seagull_optimization_algorithm.parameters.common_parameters import CommonParameters


class GaParameters:

    def __init__(self, common_parameters: CommonParameters, is_binary: bool):
        self.common_parameters = common_parameters
        self.is_binary = is_binary
