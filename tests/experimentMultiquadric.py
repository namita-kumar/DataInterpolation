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

########################## Hardy Multiquadric tests ###########################
sampledValue = TestFunctions.franke_function_wBias(xData)
evalPoints = space.rvs(n_evalPoints)
exactValue = TestFunctions.franke_function_wBias(evalPoints)

err = []
for ndx in range(100):
    randpar = random.random()
    obj = Interpolator(xData, sampledValue, data = xData, par=[randpar, 1.0], order=1, type="Hardy multiquadric")
    functionCoeff, approxValue, rmsError = obj.interpolate_evaluate(evalPoints, exactValue)
    err.append(rmsError)
    print("Error with parameter ", randpar," is", "{:.3e}".format(rmsError))
    print("The coefficients of (1,x,y):", functionCoeff[-3::])

fig = plt.figure()
plt.hist(err, range=(0,10))
fig.show()
fig2 = plt.figure()
plt.hist(err, range=(0,100))
plt.show()
