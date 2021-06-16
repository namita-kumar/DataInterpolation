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
n_evalPoints = 10

space = Space([(0, 1.0), (0, 1.0)])

halton = Halton()
xData = halton.generate(space, n_samples)
sampledValue = TestFunctions.franke_function(xData)
plot_3Dspace(xData, sampledValue, 'Cyclic Product')

# dMatrix = Interpolator.rbf_dist_matrix(xData, xData, 445.21, type="gaussian")
# functionCoeff = np.matmul(np.linalg.inv(dMatrix), sampledValue)
dMatrix = Interpolator.rbf_dist_matrix(xData, xData, 1.0, 1.0, type="hardy")
functionCoeff = np.matmul(np.linalg.inv(dMatrix), sampledValue)

# randomly seelcted evaluation points
evalPoints = space.rvs(n_evalPoints)
exactValue = TestFunctions.franke_function(evalPoints)
# eMatrix = Interpolator.rbf_dist_matrix(evalPoints, xData, 445.21, type="gaussian")
eMatrix = Interpolator.rbf_dist_matrix(evalPoints, xData, 1.0, 1.0, type="hardy")
approxValue = np.matmul(eMatrix, functionCoeff)
rmsError = Interpolator.rms_error(exactValue, approxValue)
pdb.set_trace()
