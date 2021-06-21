import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Halton
import pdb
from test_functions import TestFunctions
from interpolator import Interpolator


def plot_3Dspace(x, fValue, title):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.array(x)[:, 0], np.array(x)[:, 1], fValue, 'bo')
    ax.set_xlabel("X1")
    ax.set_xlim([0, 1])
    ax.set_ylabel("X2")
    ax.set_ylim([0, 1])
    plt.title(title)


n_samples = 1089
n_evalPoints = 100

space = Space([(0, 1.0), (0, 1.0)])

halton = Halton()
xData = halton.generate(space, n_samples)
sampledValue = TestFunctions.franke_function(xData)
evalPoints = space.rvs(n_evalPoints)
exactValue = TestFunctions.franke_function(evalPoints)
plot_3Dspace(xData, sampledValue, 'Franke function')

listRBF = ["gaussian", "hardy multiquadric", "thin plate splines"]
par = [[445.21], [1.0, 1.0],[]]
parLB = [[445.21], [1.0, 1.0]]
parUB = [[445.21], [1.0, 1.0]]

for ndx in range(len(listRBF)):
    functionCoeff, approxValue, rmsError = Interpolator.interpolate_evaluate(xData, xData, sampledValue, evalPoints, exactValue, par[ndx], type=listRBF[ndx])
    print("Error under", listRBF[ndx], ":", rmsError)
