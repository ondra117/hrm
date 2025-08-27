from tqdm import tqdm
import time
import numpy as np
from scipy.special import erf, erfinv


def lacum_normal(x, a):
    return erfinv(2 * x * erf(a) - erf(a))


a = np.random.uniform(0, 1, 100)

b = lacum_normal(a, 1)

print(np.min(b))
print(np.max(b))
