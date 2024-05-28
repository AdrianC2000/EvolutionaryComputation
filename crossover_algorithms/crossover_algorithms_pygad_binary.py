import numpy as np


class CrossoverAlgorithmsPyGADBinary:
    def __init__(self, binary_chain_length: int):
        self.__binary_chain_length = binary_chain_length
        self.__METHODS = {
            "three-point": self._three_point_crossover,
            "grain": self._grain_crossover,
            "mssx": self._mssx_crossover,
            "three-parent": self._three_parent_crossover,
            "nonuniform": self._nonuniform_crossover,
        }

    def get_methods(self):
        return self.__METHODS

    def _three_point_crossover(self, parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size, dtype=int)
        for k in range(offspring_size[0] - 1):
            crossover_points = sorted(np.random.choice(range(1, self.__binary_chain_length), 3, replace=False))
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            parent_a = parents[parent1_idx, :]
            parent_b = parents[parent2_idx, :]

            descendant_1, descendant_2 = parent_a[:], parent_b[:]
            descendant_1[crossover_points[0]:crossover_points[1]] = parent_b[crossover_points[0]:crossover_points[1]]
            descendant_2[crossover_points[0]:crossover_points[1]] = parent_a[crossover_points[0]:crossover_points[1]]

            descendant_1[crossover_points[2]:] = parent_b[crossover_points[2]:]
            descendant_2[crossover_points[2]:] = parent_a[crossover_points[2]:]

            offspring[k, :] = descendant_1
            offspring[k + 1, :] = descendant_2

        return offspring

    def _grain_crossover(self, parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size, dtype=int)
        for k in range(offspring_size[0]):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            parent_a = parents[parent1_idx, :]
            parent_b = parents[parent2_idx, :]

            descendant_1 = parent_a[:]

            for i in range(0, self.__binary_chain_length):
                if np.random.random() > 0.5:
                    descendant_1[i] = parent_b[i]
            offspring[k, :] = descendant_1

        return offspring

    def _mssx_crossover(self, parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size, dtype=int)
        for k in range(offspring_size[0] - 1):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            parent_a = parents[parent1_idx, :]
            parent_b = parents[parent2_idx, :]

            parent_a_combined = np.array(list(self._combine_parent(parent_a)), dtype=int)
            parent_b_combined = np.array(list(self._combine_parent(parent_b)), dtype=int)

            sexes_count = 2
            n = len(parent_a_combined)
            offsprings = np.zeros((2, n)).astype(int)
            rng = np.random.default_rng()

            parents = np.vstack((parent_a_combined, parent_b_combined))

            for offspring_count in range(offsprings.shape[0]):
                for i in range(n):
                    random_sex = rng.integers(sexes_count)
                    offsprings[offspring_count][i] = parents[random_sex][i]

            descendant_1 = self._combine_parent(offsprings[0])
            descendant_2 = self._combine_parent(offsprings[1])

            offspring[k, :] = np.array(list(descendant_1))
            offspring[k + 1, :] = np.array(list(descendant_2))

        return offspring

    def _three_parent_crossover(self, parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size, dtype=int)
        for k in range(offspring_size[0] - 2):
            points = np.sort(np.random.choice(range(1, self.__binary_chain_length), 2, replace=False))
            first_point, second_point = points

            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            parent3_idx = (k + 2) % parents.shape[0]
            parent_a = parents[parent1_idx, :]
            parent_b = parents[parent2_idx, :]
            parent_c = parents[parent3_idx, :]

            descendant_1 = np.concatenate((parent_a[:first_point], parent_b[first_point:]))
            descendant_2 = np.concatenate((parent_b[:first_point], parent_a[first_point:]))
            descendant_3 = np.concatenate((descendant_1[:second_point], parent_c[second_point:]))

            offspring[k, :] = descendant_1
            offspring[k + 1, :] = descendant_2
            offspring[k + 2, :] = descendant_3

        return offspring

    def _nonuniform_crossover(self, parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size, dtype=int)
        for k in range(offspring_size[0]):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            parent_a = parents[parent1_idx, :]
            parent_b = parents[parent2_idx, :]

            descendant_1 = parent_a[:]

            for i in range(0, self.__binary_chain_length):
                if np.random.random() > np.random.random():
                    descendant_1[i] = parent_b[i]
            offspring[k, :] = descendant_1

        return offspring

    @staticmethod
    def _combine_parent(parent: np.ndarray) -> str:
        return "".join(str(element) for element in parent)

    def _map_descendant(self, descendant: str) -> np.ndarray:
        return np.array([descendant[i:i + self.__binary_chain_length]
                         for i in range(0, len(descendant), self.__binary_chain_length)])
