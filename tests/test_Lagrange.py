import numpy as np
import matplotlib.pyplot as plt
from skopt.sampler.halton import Halton
from skopt.space import Space
from skopt.sampler import Grid
import pdb
from dataInterpolation.testFunctions import TestFunctions
from dataInterpolation.interpolator import Interpolator
import time

if __name__ == "__main__":
    n_samples = 100
    n_evalPoints = 5

    space = Space([(0.0, 1.0), (0.0, 1.0)])

    par = [1.0]
    type = 'Thin plate splines'
    order = 1

    grid = Grid()
    xData = grid.generate(space, n_samples)
    # fillDistance, separationRadius = Utilities.fill_separation_distance(xData)
    # print(fillDistance)
    # sampledValue = TestFunctions.wedland22_function(np.array(xData), [0,0])
    sampledValue = TestFunctions.cyclic_product(np.array(xData))
    halton = Halton()
    evalPoints = halton.generate(space, n_evalPoints)
    # exactValue = TestFunctions.wedland22_function(np.array(evalPoints), [0,0])
    exactValue = TestFunctions.cyclic_product(np.array(evalPoints))
    #############       Lagrange Interpolant Parallel       #############
    obj = Interpolator(xData, sampledValue, par=par, order=order, type=type)
    # footprint = np.abs(7*fillDistance*np.log(fillDistance))
    # n_neigbors = int(10*(np.log10(n_samples))**2)
    n_neigbors = 200
    print("Number of neighbors", n_neigbors)
    startTime = time.time()
    print("Start", startTime)
    lagrangeFname = obj.largrange_interpolant_parallel(n_neigbors)
    print("Time taken to find Lagrange functions", time.time() - startTime)
    # lagrangeFname = 'Lagrange_Thin plate splinesOrder1_10000Centers.csv'
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

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 12), subplot_kw={'projection': '3d'})
    ax1.plot_trisurf(np.array(xData)[:,0], np.array(xData)[:,1], sampledValue, cmap='viridis', edgecolor='none')
    errorVal = np.absolute(fVal - exactValue)
    ax2.plot_trisurf(np.array(evalPoints)[:,0], np.array(evalPoints)[:,1], fVal, cmap='viridis', edgecolor='none')
    plt.tight_layout()
    fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(8, 12), subplot_kw={'projection': '3d'})
    ax4.plot_trisurf(np.array(evalPoints)[:,0], np.array(evalPoints)[:,1], errorVal, cmap='viridis', edgecolor='none')
    plt.tight_layout()
    
    obj1 = Interpolator(xData, sampledValue, par=par, order=order, type=type)
    startTime = time.time()
    print("Start", startTime)
    lagrangeFname = obj1.largrange_interpolant()
    print("Time taken to find Lagrange functions", time.time() - startTime)
    # lagrangeFname = 'Lagrange_Thin plate splinesOrder1_10000Centers.csv'
    fVal, rmsError = Interpolator.largrange_evaluate(xData, sampledValue, evalPoints, lagrangeFname, exactValue, par, order=order, type=type)
    print('RMS approximation error = ', rmsError)

    ax3.plot_trisurf(np.array(evalPoints)[:,0], np.array(evalPoints)[:,1], fVal, cmap='viridis', edgecolor='none')
    plt.tight_layout()
    errorVal = np.absolute(fVal - exactValue)
    ax5.plot_trisurf(np.array(evalPoints)[:,0], np.array(evalPoints)[:,1], errorVal, cmap='viridis', edgecolor='none')
    plt.tight_layout()
    plt.show()
