import numpy as np
from numpy import ndarray


class FitnessFunction:

    @staticmethod
    def fitness_function1(individual: ndarray) -> float:
        return float(np.sum(np.power(individual, 2)))
    
    # katsuura
    @staticmethod
    def fitness_function(x: ndarray):
        sum_rana = 0
        
        for i in range(len(x) - 1):
            sqrt_term1 = np.sqrt(np.abs(x[i + 1] + x[i] + 1))
            sqrt_term2 = np.sqrt(np.abs(x[i + 1] - x[i] + 1))
            
            term1 = x[i] * np.cos(sqrt_term1) * np.sin(sqrt_term2)
            term2 = (1 + x[i + 1]) * np.sin(sqrt_term1) * np.cos(sqrt_term2)
            
            sum_rana += term1 + term2
            
        return sum_rana
    
    # rana
    @staticmethod
    def fitness_function2(x):
        D = len(x)
        term = 1.0 * 10/D**2
        for i in range(D):
            inner_sum = 0.0
            for j in range(1, 33):
                term_j = 2**j * x[i]
                inner_sum += np.abs(term_j - np.round(term_j)) / (2**j)
            term *= (1 + (i + 1) * inner_sum) ** (10.0 / D**1.2)
        return term - 10/D**2
