import numpy as np
import matplotlib.pyplot as plt
from skopt.sampler.halton import Halton
from skopt.space import Space
from skopt.sampler import Grid
import pdb
from dataInterpolation.testFunctions import TestFunctions
from dataInterpolation.interpolator import Interpolator
from dataInterpolation.utilities import Utilities
from dataInterpolation.latticelagrange import LatticeLagrangeInterpolator
import time

if __name__ == "__main__":
    n_samples = 3000
    n_evalPoints = 5

    space = Space([(-1.2, 1.2), (-1.2, 1.2)])

    par = [1.0]
    type = 'Thin plate splines'
    order = 1

    grid = Grid()
    centers = grid.generate(space, n_samples)
    fillDistance, separationRadius = Utilities.fill_separation_distance(centers)
    #override fillDistance
    fillDistance = 0.2
    print("Fill distance: ", fillDistance)
    noOfNeighbors = int(np.abs(np.log(fillDistance)))
    footprint = np.abs(7*fillDistance*np.log(fillDistance))
    sampledValue = TestFunctions.wedland22_function(np.array(centers), [0,0])
    interpolate = LatticeLagrangeInterpolator([-1,1], fillDistance)
    Lagrange_00_Coeff_fname = interpolate.lagrange_function_0_0()
    print(Lagrange_00_Coeff_fname)

    
