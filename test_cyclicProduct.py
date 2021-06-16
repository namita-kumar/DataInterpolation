import numpy as np
import matplotlib.pyplot as plt
from test_functions import TestFunctions
from skopt.space import Space
from skopt.sampler import Sobol
from skopt.sampler import Lhs
from skopt.sampler import Halton
from skopt.sampler import Hammersly
from skopt.sampler import Grid
from scipy.spatial.distance import pdist
import pdb


def plot_3Dspace(x, fValue, title):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.array(x)[:, 0], np.array(x)[:, 1], fValue, 'bo')
    ax.set_xlabel("X1")
    ax.set_xlim([0, 1])
    ax.set_ylabel("X2")
    ax.set_ylim([0, 1])
    plt.title(title)


n_samples = 289

space = Space([(0, 1.0), (0, 1.0)])

halton = Halton()
x = halton.generate(space, n_samples)
fValue = TestFunctions.cyclic_product(0, 1.0, x)
plot_3Dspace(x, fValue, 'Cyclic Product')
plt.show()
