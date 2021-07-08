import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Halton
import pdb
from test_functions import TestFunctions
from interpolator import Interpolator
import time


def plot_space(x, fValue, title):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.array(x)[:, 0], np.array(x)[:, 1], fValue, 'bo')
    ax.set_xlabel("X1")
    ax.set_xlim([0, 1])
    ax.set_ylabel("X2")
    ax.set_ylim([0, 1])
    plt.title(title)
    return plt


if __name__ == "__main__":
    n_samples = 3000
    n_evalPoints = 100

    space = Space([(0, 1.0), (0, 1.0)])

    par = [1.0]
    type = "Thin plate splines"
    order = 1

    halton = Halton()
    xData = halton.generate(space, n_samples)
    sampledValue = TestFunctions.franke_function_wBias(xData)
    evalPoints = space.rvs(n_evalPoints)
    exactValue = TestFunctions.franke_function_wBias(evalPoints)
    # pdb.set_trace()
    ###########################     Lagrange Interpolant       ################
    startTime = time.time()
    print("Start", startTime)
    # lagrangeFname = Interpolator.largrange_interpolant_parallel(xData, sampledValue, par, order=order, type=type)
    lagrangeFname = Interpolator.largrange_interpolant(xData, sampledValue, par, order=order, type=type)
    print("End", time.time() - startTime)
    # lagrangeFname = 'Lagrange_GaussianOrder1_2000Centers.csv'
    fVal, rmsError = Interpolator.largrange_evaluate(xData, sampledValue, evalPoints, lagrangeFname, exactValue, par, order=order, type=type)
    print('RMS approximation error = ', rmsError)

    fig0 = plt.figure()
    plt.scatter(np.array(xData)[:, 0], np.array(xData)[:, 1], c='blue', marker='o')
    plt.scatter(np.array(evalPoints)[:, 0], np.array(evalPoints)[:, 1], c='red', marker='x')
    plt.legend(["Centers","Evaluated Points"])
    plt.xlabel("X1")
    plt.xlim([0, 1])
    plt.ylabel("X2")
    plt.ylim([0, 1])

    fig3D, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 12), subplot_kw={'projection': '3d'})
    ax1.plot_trisurf(np.array(xData)[:,0], np.array(xData)[:,1], sampledValue, cmap='viridis', edgecolor='none');
    ax2.plot_trisurf(np.array(evalPoints)[:,0], np.array(evalPoints)[:,1], fVal, cmap='viridis', edgecolor='none');
    plt.tight_layout()
    plt.show()
