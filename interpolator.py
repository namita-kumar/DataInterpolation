import numpy as np
from scipy.spatial import distance_matrix
import pdb


class Interpolator:
    @staticmethod
    def interpolate_evaluate(xData, centers, sampledValue, evalPoints, exactValue, par, parLB=None, parUB=None, testFunction=None, type=None):
        functionCoeff = Interpolator.interpolate(xData, centers, sampledValue, par, parLB, parUB, testFunction, type)
        approxValue, rmsError = Interpolator.evaluate(evalPoints, functionCoeff, exactValue, xData, par, type)
        return functionCoeff, approxValue, rmsError

    @staticmethod
    def interpolate(xData, centers, sampledValue, par, parLB=None, parUB=None, testFunction=None, type=None):
        dMatrix = Interpolator.rbf_dist_matrix(xData, centers, par, type)
        functionCoeff = np.matmul(np.linalg.inv(dMatrix), sampledValue)
        return functionCoeff

    @staticmethod
    def rbf_dist_matrix(data, centers, par, type=None):
        distanceMatrix = distance_matrix(data, centers)
        if type == "gaussian":
            distanceMatrix = np.exp(-par[0]*np.power(distanceMatrix, 2))
        elif type == "hardy multiquadric":
            distanceMatrix = np.power(distanceMatrix, 2) + par[0]**2*np.ones(np.shape(distanceMatrix))
            distanceMatrix = (-1)**(np.ceil(par[1]))*np.power(distanceMatrix, par[1]/2.0)
        elif type == "thin plate splines":
            distanceMatrix = np.power(distanceMatrix, 2)*np.log(distanceMatrix + 1e-10*np.ones(np.shape(distanceMatrix)))
        return distanceMatrix

    @staticmethod
    def evaluate(evalPoints, functionCoeff, exactValue, xData, par, type):
        eMatrix = Interpolator.rbf_dist_matrix(evalPoints, xData, par, type)
        approxValue = np.matmul(eMatrix, functionCoeff)
        rmsError = np.sqrt(((exactValue - approxValue) ** 2).mean())
        return approxValue, rmsError
