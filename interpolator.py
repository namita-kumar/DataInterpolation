import numpy as np
from scipy.spatial import distance_matrix
import pdb


class Interpolator:
    @staticmethod
    def rbf_dist_matrix(data, centers, *args, type=None):
        distanceMatrix = distance_matrix(data, centers)
        if type == "gaussian":
            distanceMatrix = np.exp(-args[0]*np.power(distanceMatrix, 2))
        elif type == "hardy":
            distanceMatrix = np.power(distanceMatrix, 2) + args[0]*np.ones(np.shape(distanceMatrix))
            distanceMatrix = - np.power(distanceMatrix, args[1]/2.0)
        return distanceMatrix

    @staticmethod
    def rms_error(exactValue, approxValue):
        return np.sqrt(((exactValue - approxValue) ** 2).mean())
