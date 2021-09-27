import numpy as np
from scipy.spatial import Voronoi, distance_matrix
from math import factorial


class Utilities:

    @staticmethod
    def rms_error(value1, value2):
        ''' Calcualted root mean square error between two arrays value1 and value2
            Args:
                value1 (list): array #1
                value2 (list): array #2
            Returns:
                rmsError (float): error between the two arrays
        '''
        rmsError = np.sqrt(((value1 - value2) ** 2).mean())
        return rmsError

    @staticmethod
    def n_choose_r(n, r):
        ''' Returns n choose r
            Args:
            n (float): Number to choose from
            r (float): Number of choices
            Returns:
            n_choose_r (float): factorial(n)/factorial(r)/factorial(n-r)
        '''
        return factorial(n)/factorial(r)/factorial(n-r)

    @staticmethod
    def fill_separation_distance(points):
        ''' Retruns the fill distance and separation radius of a set of points
        '''
        voronoiVertices = Voronoi(points)
        voronoiDistanceMatrix = distance_matrix(voronoiVertices.vertices, points)
        selfDistanceMatrix = distance_matrix(points, points)
        np.fill_diagonal(selfDistanceMatrix, np.inf)
        fillDistance = np.max(voronoiDistanceMatrix.min(axis=0))/2.0
        separationRadius = np.min(selfDistanceMatrix)/2.0
        meshRatio = fillDistance/separationRadius
        return fillDistance, separationRadius