import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Halton
import random
import pdb
from dataInterpolation.testFunctions import TestFunctions
from dataInterpolation.interpolator import Interpolator

n_samples = 1089
n_evalPoints = 100

space = Space([(0, 1.0), (0, 1.0)])

halton = Halton()
xData = halton.generate(space, n_samples)

###########################     Order 1 tests       ############################
sampledValue = TestFunctions.franke_function_wBias(xData)
evalPoints = space.rvs(n_evalPoints)
exactValue = TestFunctions.franke_function_wBias(evalPoints)

# Euclidean distance matrix is the default
listRBF = ["Euclidean", "Gaussian", "Hardy multiquadric", "Thin plate splines"]
par = [[], [445.21], [0.08708, 1.0], [1.0]]

for ndx in range(len(listRBF)):
    obj = Interpolator(xData, sampledValue, data = xData, par= par[ndx], order=1, type=listRBF[ndx])
    functionCoeff, approxValue, rmsError = obj.interpolate_evaluate(evalPoints, exactValue)
    if rmsError >=1.0:
        print("ERROR: RMS error too high with ", listRBF[ndx])
        exit()
    else:
        print("PASSED ", listRBF[ndx], "distance matrix with RMS error:", "{:.3e}".format(rmsError))
        print("The coefficients of (1,x,y):", functionCoeff[-3::])
