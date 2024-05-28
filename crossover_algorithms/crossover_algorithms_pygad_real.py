import random
from typing import Tuple
import numpy as np
from numpy import ndarray
from fitness_functions import FitnessFunction


class CrossoverAlgorithmsPyGADReal:
    def __init__(self, bounds: Tuple[int, int], is_min_searched: bool = False, fitness_function: str = "square_sum"):
        self.__METHODS = {
            "arithmetical": self._arithmetical_crossover,
            "linear": self._linear_crossover,
            "blend_alpha": self._blend_alpha_crossover,
            "blend_alpha_and_beta": self._blend_alpha_and_beta_crossover,
            "average": self._average_crossover,
            "unfair_average": self._unfair_average_crossover,
            "gaussian_uniform": self._gaussian_uniform_crossover,
        }
        self.__bounds = bounds
        self.__is_min_searched = is_min_searched
        self.__fitness_function = fitness_function

    def get_methods(self):
        return self.__METHODS

    @staticmethod
    def _arithmetical_crossover(parents, offspring_size, ga_instance) -> ndarray:
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0] - 1):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            parent_a = parents[parent1_idx, :]
            parent_b = parents[parent2_idx, :]

            alpha = np.random.rand()
            child_1 = alpha * parent_a + (1 - alpha) * parent_b
            child_2 = alpha * parent_b + (1 - alpha) * parent_a
            offspring[k, :] = child_1
            offspring[k + 1, :] = child_2

        return offspring

    def _linear_crossover(self, parents, offspring_size, ga_instance) -> ndarray:
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0] - 1):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            parent_a = parents[parent1_idx, :]
            parent_b = parents[parent2_idx, :]

            if np.array_equal(parent_a, parent_b):
                offspring[k, :] = parent_a
                offspring[k + 1, :] = parent_b
                continue

            child_z = 0.5 * parent_a + 0.5 * parent_b
            child_v = 1.5 * parent_a - 0.5 * parent_b
            child_w = -0.5 * parent_a + 1.5 * parent_b

            fitness_function = FitnessFunction(self.__fitness_function).selected_function

            results = {
                tuple(child_z): fitness_function(child_z),
                tuple(child_v): fitness_function(child_v),
                tuple(child_w): fitness_function(child_w)
            }
            max_results = sorted(results.items(), key=lambda x: x[1], reverse=not self.__is_min_searched)

            if len(max_results) != 3:
                offspring[k, :] = parent_a
                offspring[k + 1, :] = parent_b
                continue

            child_a = np.copy(parent_a)
            child_b = np.copy(parent_b)
            best_child_a = np.array(max_results[0][0])
            best_child_b = np.array(max_results[1][0])
            best_child_backup = np.array(max_results[2][0])

            if self._individual_in_bounds(best_child_a):
                child_a = best_child_a
            elif self._individual_in_bounds(best_child_backup):
                child_a = best_child_backup
            if self._individual_in_bounds(best_child_b):
                child_b = best_child_b

            offspring[k, :] = child_a
            offspring[k + 1, :] = child_b
        return offspring

    def _blend_alpha_crossover(self, parents, offspring_size, ga_instance) -> ndarray:
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0] - 1):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            parent_x = parents[parent1_idx, :]
            parent_y = parents[parent2_idx, :]

            x_child = np.copy(parent_x)
            y_child = np.copy(parent_y)

            for index, (x, y) in enumerate(zip(parent_x, parent_y)):
                distance = abs(x - y)
                alpha = np.random.rand()
                u1 = min(x, y) - alpha * distance
                u2 = max(x, y) + alpha * distance

                new_x = random.uniform(u1, u2)
                new_y = random.uniform(u1, u2)

                self.update_kids(index, new_x, new_y, x_child, y_child)

            offspring[k, :] = x_child
            offspring[k + 1, :] = y_child

        return offspring

    def _blend_alpha_and_beta_crossover(self, parents, offspring_size, ga_instance) -> ndarray:
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0] - 1):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            parent_x = parents[parent1_idx, :]
            parent_y = parents[parent2_idx, :]

            fitness_function = FitnessFunction(self.__fitness_function).selected_function

            if (self.__is_min_searched and fitness_function(parent_x) > fitness_function(parent_y)) or (
                    not self.__is_min_searched and fitness_function(parent_x) < fitness_function(parent_y)):
                parent_x, parent_y = parent_y, parent_x

            x_child = np.copy(parent_x)
            y_child = np.copy(parent_y)

            for index, (x, y) in enumerate(zip(parent_x, parent_y)):
                distance = abs(x - y)
                alpha = np.random.rand()
                beta = np.random.rand()
                if x <= y:
                    u1 = x - alpha * distance
                    u2 = y + beta * distance
                    new_x = random.uniform(u1, u2)
                    new_y = random.uniform(u1, u2)

                    self.update_kids(index, new_x, new_y, x_child, y_child)
                else:
                    u1 = y - beta * distance
                    u2 = y + alpha * distance

                    new_x = random.uniform(u1, u2)
                    new_y = random.uniform(u1, u2)

                    self.update_kids(index, new_x, new_y, x_child, y_child)

            offspring[k, :] = x_child
            offspring[k + 1, :] = y_child

        return offspring

    def _average_crossover(self, parents, offspring_size, ga_instance) -> ndarray:
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0] - 1):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            parent_a = parents[parent1_idx, :]
            parent_b = parents[parent2_idx, :]

            child_a = np.copy(parent_a)
            child_b = np.copy(parent_b)

            new_child_a = (parent_a + parent_b) / 2

            if self._individual_in_bounds(new_child_a):
                child_a = new_child_a
                new_child_b = (new_child_a + parent_b) / 2
                if self._individual_in_bounds(new_child_b):
                    child_b = new_child_b
            offspring[k, :] = child_a
            offspring[k + 1, :] = child_b

        return offspring

    def _unfair_average_crossover(self, parents, offspring_size, ga_instance) -> ndarray:
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0] - 1):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            X_parent = parents[parent1_idx, :]
            Y_parent = parents[parent2_idx, :]

            c = 2

            if np.array_equal(X_parent, Y_parent):
                offspring[k, :] = X_parent
                offspring[k + 1, :] = Y_parent
                continue

            n = X_parent.size
            X_child = np.copy(X_parent)
            Y_child = np.copy(Y_parent)

            alpha = np.random.rand()

            Z = np.zeros(c, dtype=float)
            W = np.zeros(c, dtype=float)

            Q = np.zeros(n - c, dtype=float)
            V = np.zeros(n - c, dtype=float)

            for i in range(c):
                Z[i] = 1 + (1 / alpha) * X_parent[i] - (1 / alpha) * Y_parent[i]
                W[i] = 1 - (1 / alpha) * X_parent[i] + (1 / alpha) * Y_parent[i]

            for index, i in enumerate(range(c, n)):
                Q[index] = -(1 / alpha) * X_parent[i] + 1 + (1 / alpha) * Y_parent[i]
                V[index] = (1 / alpha) * X_parent[i] + 1 - (1 / alpha) * Y_parent[i]

            X_child_new = np.concatenate((Z, Q))
            Y_child_new = np.concatenate((W, V))

            if self._individual_in_bounds(X_child_new):
                X_child = X_child_new
            if self._individual_in_bounds(Y_child_new):
                Y_child = Y_child_new

            offspring[k, :] = X_child
            offspring[k + 1, :] = Y_child

        return offspring

    def _gaussian_uniform_crossover(self, parents, offspring_size, ga_instance) -> ndarray:
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0] - 1):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            x_parent = parents[parent1_idx, :]
            y_parent = parents[parent2_idx, :]

            x_child = np.copy(x_parent)
            y_child = np.copy(y_parent)
            for index, (x, y) in enumerate(zip(x_parent, y_parent)):
                distance = abs(x - y)
                random_number = np.random.rand()
                alpha = np.random.normal(0, 1)
                if random_number <= 0.5:
                    new_x = x + alpha * distance / 3
                    new_y = y + alpha * distance / 3
                else:
                    new_x = y + alpha * distance / 3
                    new_y = x + alpha * distance / 3
                self.update_kids(index, new_x, new_y, x_child, y_child)

            offspring[k, :] = x_child
            offspring[k + 1, :] = y_child

        return offspring

    def _individual_in_bounds(self, individual: ndarray):
        return np.all((individual >= self.__bounds[0]) & (individual <= self.__bounds[1]))

    def update_kids(self, index, new_x, new_y, x_child, y_child):
        if self.__bounds[0] <= new_x <= self.__bounds[1]:
            x_child[index] = new_x
        if self.__bounds[0] <= new_y <= self.__bounds[1]:
            y_child[index] = new_y
