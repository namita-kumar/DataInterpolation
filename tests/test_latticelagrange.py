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
    n_evalPoints = 100

    evalspace = Space([(-1.0, 0.5), (-1.0, 0.5)])

    par = [1.0]
    type = 'Gaussian'
    order = 1

    grid = Grid()
    # centers = grid.generate(space, n_samples)
    evalPoints = grid.generate(evalspace, n_evalPoints)
    # desired fillDistance
    fillDistance = 0.2
    print("Fill distance: ", fillDistance)
    noOfNeighbors = int(np.abs(np.log(fillDistance)))
    footprint = np.abs(7*fillDistance*np.log(fillDistance))
    # sampledValue = TestFunctions.wedland22_function(np.array(centers), [0,0])
    # interpolate = LatticeLagrangeInterpolator([-1,1], fillDistance, par =[0.08708, 1.0], type = "Hardy multiquadric")
    interpolate = LatticeLagrangeInterpolator([-1,1], fillDistance, par =[0.0], type = "Bessel")
    fname = interpolate.lagrange_function_0_0()
    # fname = "dataDump\LatticeLagrange_Thin plate splines_4FootPrintPoints.csv"
    f_val = interpolate.eval_lagrange_function_0_0([[-0.1999,-0.1999]],fname)

    print(f_val)

    
