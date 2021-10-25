import numpy as np
from scipy.spatial import Voronoi, distance_matrix
from math import factorial
import itertools


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
    
    @staticmethod
    def generate_lattice(pointOfInterest,footprintRadius,fillDistance):
        x = np.arange(pointOfInterest[0]-footprintRadius, pointOfInterest[0]+footprintRadius+fillDistance, fillDistance)
        y = np.arange(pointOfInterest[1]-footprintRadius, pointOfInterest[1]+footprintRadius+fillDistance, fillDistance)
        centers = list(itertools.product(x, y))
        return centers
    
    @staticmethod
    def dft_2D(y,m):
        '''
            Returns 2D DFT of y.
            Args:
                y (2D list): the distance matrix of which we want the Fourier transform
                m (int): number of points radially from the center of the lattice 
            Returns:
                y_hat (list of float): DFT of y for every lattice center 
        '''
        numOfPOints = (2*m+1)**2
        y_hat = []
        # number of points in each direction
        n = 2*m+1
        # calculate fourier transform for each point in the lattice
        for ndx in range(numOfPOints):
            # the point of interest is ndx. Convert the index ndx to indices (r,s) where s is the indexing in x-direciton and r is the indexing in y direction
            r = int(ndx/m)
            s = int(ndx - r*m)
            y_hat_j_k = 0
            for j in range(n):
                for k in range(n):
                    y_hat_j_k += y[ndx,(j*m+k)]*np.exp((-2*np.pi*1j/n)*(r*j+s*k))
            y_hat.append(y_hat_j_k)
        return np.array(y_hat)

    @staticmethod
    def dft_inverse_2D(y_hat,m):
        '''
            Returns 2D inverse DFT of y_hat.
            Args:
                y_hat (numpy list): list of fourier transforms
                m (int): number of points radially from the center of the lattice 
            Returns:
                y (float): inverse DFT of y_hat. Gives the lagrnage coefficients for every lattice center 
        '''
        numOfPOints = (2*m+1)**2
        # number of points in each direction
        n = 2*m+1
        y = []
        for j in range(numOfPOints):
            y_j = []
            if j==1:
                y_j = [y[0]]
            elif j>1:
                y_j = np.transpose(y)[j,:]
            for k in range(j,numOfPOints):
                y_j_k = 0
                print("Column number = ",k)
                for ndx in range(numOfPOints):
                    # the point of interest is ndx. Convert the index ndx to indices (r,s) where s is the indexing in x-direciton and r is the indexing in y direction
                    r = int(ndx/m)
                    s = int(ndx - r*m)
                    y_j_k += y_hat[ndx]*np.exp((2*np.pi*1j/n)*(r*j+s*k))
                    y_j_k = np.abs(y_j_k)
                y_j_k = y_j_k/(n**2)
                y_j = np.concatenate((y_j,[y_j_k]))   #append (j,k)-th coefficient 
            if len(y)==0:
                y = np.array(y_j)
            else:
                y = np.vstack((y, np.array(y_j)))
            print("Row number = ",j)
        return y