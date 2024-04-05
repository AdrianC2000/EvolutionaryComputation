import numpy as np

sexes_count = 2
n = 10

rng = np.random.default_rng()
sexes = np.arange(sexes_count)
parents = np.zeros((sexes_count, n + 1)).astype(int)

for sex in sexes:
    parents[sex] = np.concatenate([rng.integers(0, 2, n), [sex]])

print(f'Parents: \n{parents}')

heritage = np.zeros(sexes_count).astype(int)
offspring = np.zeros(n + 1).astype(int)

for i in range(n):
    random_sex = rng.integers(sexes_count)
    offspring[i] = parents[random_sex][i]
    heritage[random_sex] += 1

max_ind = np.where(heritage == np.max(heritage))[0]
offspring[-1] = rng.choice(max_ind)
print(f'Offspring: {offspring}')
