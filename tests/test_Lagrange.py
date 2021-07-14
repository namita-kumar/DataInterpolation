import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Grid
import pdb
from dataInterpolation.testFunctions import TestFunctions
from dataInterpolation.interpolator import Interpolator
import time

if __name__ == "__main__":
    n_samples = 1000
    n_evalPoints = 100

    space = Space([(0, 1.0), (0, 1.0)])

    par = [1.0]
    type = "Thin plate splines"
    order = 1

    grid = Grid()
    xData = grid.generate(space, n_samples)
    sampledValue = TestFunctions.franke_function_wBias(xData)
    evalPoints = space.rvs(n_evalPoints)
    exactValue = TestFunctions.franke_function_wBias(evalPoints)
    ###########################     Lagrange Interpolant       ################
    startTime = time.time()
    print("Start", startTime)
    lagrangeFname = Interpolator.largrange_interpolant(xData, sampledValue, par, order=order, type=type)
    print("Time taken to find Lagrange functions", time.time() - startTime)
    # lagrangeFname = 'Lagrange_GaussianOrder1_2000Centers.csv'
    fVal, rmsError = Interpolator.largrange_evaluate(xData, sampledValue, evalPoints, lagrangeFname, exactValue, par, order=order, type=type)
    print('RMS approximation error = ', rmsError)

    fig0 = plt.figure()
    plt.scatter(np.array(xData)[:, 0], np.array(xData)[:, 1], c='blue', marker='o')
    plt.scatter(np.array(evalPoints)[:, 0], np.array(evalPoints)[:, 1], c='red', marker='x')
    plt.legend(["Centers", "Evaluated Points"])
    plt.xlabel("X1")
    plt.xlim([0, 1])
    plt.ylabel("X2")
    plt.ylim([0, 1])

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 12), subplot_kw={'projection': '3d'})
    ax1.plot_trisurf(np.array(xData)[:,0], np.array(xData)[:,1], sampledValue, cmap='viridis', edgecolor='none');
    errorVal = np.absolute(fVal - exactValue)
    ax2.plot_trisurf(np.array(evalPoints)[:,0], np.array(evalPoints)[:,1], errorVal, cmap='viridis', edgecolor='none');
    plt.tight_layout()
    fig2, ax3 = plt.subplots(1, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})
    errorVal = np.absolute(fVal - exactValue)
    ax3.plot_trisurf(np.array(evalPoints)[:,0], np.array(evalPoints)[:,1], errorVal, cmap='viridis', edgecolor='none');
    plt.tight_layout()
    plt.show()
