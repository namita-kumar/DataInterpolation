import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Halton, Hammersly, Grid
import pdb
from interpolator import Utilities

if __name__ == "__main__":
    n_samples = np.linspace(100, 5000, 50)
    space = Space([(0, 1.0), (0, 1.0)])

    halton = Halton()
    hammersly = Hammersly()
    grid = Grid()
    samplingMethods = [halton, hammersly, grid]
    meshNorm = []
    for method in samplingMethods:
        for ndx in n_samples:
            xData = method.generate(space.dimensions, int(ndx))
            fillDistance, separationRadius = Utilities.fill_separation_distance(xData)
            meshRatio = fillDistance/separationRadius
            meshNorm.append(meshRatio)

fig, ax = plt.subplots()
ax.plot(n_samples, meshNorm[0:len(n_samples)], label='halton')
ax.plot(n_samples, meshNorm[len(n_samples):2*len(n_samples)], label='hammersly')
ax.plot(n_samples, meshNorm[2*len(n_samples)::], label='grid')
ax.set_xlabel('number of samples')
ax.set_ylabel('mesh ratio')
ax.legend()
plt.show()
