import numpy as np
from scipy.spatial import distance_matrix
import itertools
from math import factorial
from scipy.spatial import Voronoi
import csv
import pdb


class Interpolator:
    @staticmethod
    def interpolate_evaluate(xData, centers, sampledValue, evalPoints, exactValue, par, order=None, type=None):
        ''' Approxiamtes a dataset (xData, sampledValue) and determines the error in approximation at evalPoints given the exact values

            Args:
                xData (numpy array): [n,m] array where n is the dimension of the sample space and m is the number of samples.
                centers (numpy array): [n,m] array. m n-dimensional points that are the centers for the radial basis functions
                sampledValue (numpy array): the result at the sampled points
                evalPoints (numpy array): list of points where the approximated function should be evaluated
                exactValue (numpy array): the actual values at the n_evalPoints
                par (list): List of parameters corresponding to the selected radial basis function
                order (int): order of the polynomials if polynomial reproduction is desired
                type (str): Type of radail basis functions (Euclidean, Gaussian, Multiquadric, or Thin Plate Splines). Default is Euclidean

            Returns:
                functionCoeff (numpy array): list of coefficients for the selected radial basis functions
                approxValue (numpy array): the approximated function's values at evalPoints
                rmsError (float): error in approximation determined at evaPoints
        '''
        functionCoeff = Interpolator.interpolate(xData, centers, sampledValue, par, order, type)
        approxValue = Interpolator.evaluate(evalPoints, functionCoeff, centers, par, order, type)
        rmsError = Utilities.rms_error(approxValue, exactValue)
        return functionCoeff, approxValue, rmsError

    @staticmethod
    def interpolate(xData, centers, sampledValue, par, order=None, type=None):
        ''' Approxiamtes a dataset (xData, sampledValue)

            Args:
                xData (numpy array): [n,m] array where n is the dimension of the sample space and m is the number of samples.
                centers (numpy array): [n,m] array. m n-dimensional points that are the centers for the radial basis functions
                sampledValue (numpy array): the result at the sampled points
                par (list): List of parameters corresponding to the selected radial basis function
                order (int): order of the polynomials if polynomial reproduction is desired
                type (str): Type of radail basis functions (Euclidean, Gaussian, Multiquadric, or Thin Plate Splines). Default is Euclidean

            Returns:
                functionCoeff (numpy array): list of coefficients for the selected radial basis functions (and polynomials)
        '''
        # The equations are of the form [approximation matrix]*[coefficients]=[sampled values]
        # matinv is the inverse of the approximation matrix
        matinv = Interpolator.approximation_matrix(xData, centers, par, order, type, solving=1)
        RHS_value = sampledValue
        if order:
            # if polynomial reproduction is desired, add zeros to the RHS
            RHS_value = np.concatenate((RHS_value, np.zeros(Interpolator.numberOfPolyTerms)))

        functionCoeff = np.matmul(matinv, RHS_value)
        return functionCoeff

    @classmethod
    def approximation_matrix(self, xData, centers, par, order=None, type=None, solving=None):
        ''' Construct the approximation matrix
            Returns:
                inverse of approximation matrix (numpy array)
        '''
        if order:
            # if polynomial approximation is desired construct the polynomial matrix
            Interpolator.poly_matrix(xData, order)
            # get the distance matrix with radial basis functions
            Interpolator.rbf_dist_matrix(xData, centers, par, type)
            # construct the approximation matrix
            self.approxMatrix = np.column_stack((self.rbf_matrix, np.transpose(self.pMatrix)))
            if solving:
                # if solving for the function coefficients, the approximation matrix should include additional equations
                # i.e. [coefficients]*[Polynomial matrix] = 0
                # a matrix of zeroes for padding
                z_matrix = np.zeros((self.numberOfPolyTerms, self.numberOfPolyTerms))
                self.distanceMatrix = np.concatenate((self.approxMatrix, np.column_stack((self.pMatrix, z_matrix))))
                return np.linalg.inv(Interpolator.distanceMatrix)
        else:
            # no polynomial reproximation
            Interpolator.rbf_dist_matrix(xData, centers, par, type)
            self.approxMatrix = self.rbf_matrix
            if solving:
                self.distanceMatrix = self.rbf_matrix
                return np.linalg.inv(Interpolator.distanceMatrix)

    @classmethod
    def rbf_dist_matrix(self, data, centers, par, type=None):
        ''' Generates the distance matrix given a radial basis function and its parameters

            Args:
                data (numpy array): [n,m] array where n is the dimension of the sample space and m is the number of samples.
                centers (numpy array): [n,m] array. m n-dimensional points that are the centers for the radial basis functions
                par (list): List of parameters corresponding to the selected radial basis function
                type (str): Type of radail basis functions (Euclidean, Gaussian, Multiquadric, or Thin Plate Splines). Default is Euclidean

            Returns:
                distanceMatrix (numpy array): [m,m] matrix with radial basis functions evaluated with data and centers
        '''
        # default Euclidean distance matrix
        self.rbf_matrix = distance_matrix(data, centers)
        if type == "Gaussian":
            self.rbf_matrix = np.exp(-par[0]*np.power(self.rbf_matrix, 2))
        elif type == "Hardy multiquadric":
            self.rbf_matrix = np.power(self.rbf_matrix, 2) + par[0]**2*np.ones(np.shape(self.rbf_matrix))
            self.rbf_matrix = np.power(self.rbf_matrix, par[1]/2.0)
        elif type == "Thin plate splines":
            # added regularizing term to avoid taking log of zero
            self.rbf_matrix = np.power(self.rbf_matrix, 2*par[0])*np.log(self.rbf_matrix + 1e-10*np.ones(np.shape(self.rbf_matrix)))

    @classmethod
    def poly_matrix(self, data, order):
        ''' Creates a vandermonde matrix i.e. a matrix with the terms of a polynomial of n variables and m order.
            Given 2 data points with 2 variables [(x_1,y_1),(x_2,y_2)], this function Returns
            [[1, 1], [x_1, x_2], [y_1, y_2], [x_1^2, x_2^2], [y_1^2, y_2^2], [x_1*y_1, x_2*y_2]]

            Args:
                data (numpy array): [n,m] array where n is the dimension of the sample space and m is the number of samples.
                order (int): order of the polynomials

            Retuns:
                np.transpose(mat) (numpy array): [number of data points, number of polynomial terms] matrix
        '''
        n = np.shape(data)[1]
        mat = []
        for ndx in range(np.shape(data)[0]):
            # Suppose the number of variables is 2 (e.g. (x, y)) and the order is 2
            # the related polynomial terms are 1, x, y, x^2, y^2, xy
            # First: append 1. Note: this is order 0 term
            mat.extend([1])
            # Second: append terms in increasing order.
            # That is, append x, y first followed by x^2, y^2, xy
            for m in range(1, order+1):
                # Given the order of terms we are considering (i.e. m) find all corresponding combination of variables.
                # E.g. when m=2 and the variables are (x,y), itertools.product returns [(x,x),(y,y,),(x,y)]
                permute_x = itertools.product(data[ndx], repeat=m)
                # Take the product within the list [(x,x),(y,y),(x,y)] to get [x^2, y^2, xy] and then append to list
                mat.extend(list(map(np.prod, permute_x)))
        # Reshape the list to a 2D matrix. Total number of terms is given by (n+order)C(order)
        # The following gives the matrix [[1,x_1,y_1,x_1^2,y_1^2,x_1*y_1],[1,x_2,y_2,x_2^2,y_2^2,x_2*y_2], ...]
        self.numberOfPolyTerms = int(Utilities.n_choose_r(n+order, order))
        mat = np.reshape(mat, (np.shape(data)[0], self.numberOfPolyTerms))
        # We want the transpose of the above matrix
        self.pMatrix = np.transpose(mat)

    @staticmethod
    def evaluate(evalPoints, functionCoeff, centers, par, order, type):
        '''Evaluate the approximated function at specified points
            Args:
                evalPoints (numpy array): (N,M) M points of dimension M where the function should be evaluated
                functionCoeff (numpy array): coefficients of the radial basis functions (and polynomials)
                centers (numpy array): centers for the radial basis functions
                par (list): list of parameters for the radial basis function
                order (int): order of polynomials for approximation
                type (str): type of radial basis function
            Returns:
                approxValue (numpy array): the approximated function evaualted at evalPoints
        '''
        Interpolator.approximation_matrix(evalPoints, centers, par, order, type)
        approxValue = np.matmul(Interpolator.approxMatrix, functionCoeff)
        return approxValue

    @staticmethod
    def largrange_interpolant(centers, sampledValue, par, order=None, type=None):
        ''' Finds the Lagrange functions about the data points as centers. The function coefficients are written to a csv file
            Args:
                centers (numpy array): [n,m] array. m n-dimensional points that are the centers for the radial basis functions
                sampledValue (numpy array): the result at the sampled points
                par (list): List of parameters corresponding to the selected radial basis function
                order (int): order of the polynomials if polynomial reproduction is desired
                type (str): Type of radial basis functions
            Returns:
                output_fname (string): Name of the .csv file with the Lagrange function coefficients
        '''
        # Check if the mesh ratio is appropriate
        fillDistance, separationRadius = Utilities.fill_separation_distance(centers)
        meshRatio = fillDistance/separationRadius
        print("Mesh ratio=", meshRatio)
        if meshRatio>=2.5:
            print("ERROR: Mesh ratio is not suitable for accurate approximation. Try again.")
            exit()
        # Proceed to calculate and save Lagrange functions to a .csv file
        # Name of the csv file has the type of RBF, order of polynomials, and number of centers
        output_fname = 'Lagrange_' + type + 'Order' + str(order) + '_' + str(len(centers)) + 'Centers'+ '.csv'
        # Suppose we wish to find the lagrange funciton centered about k-th data point.
        # The associated equations are [approximation matrix][coefficients] = [0, 0, ... 0, 1, 0, ... 0] where 1 is in the kth position
        # Since the approximation matrix doesn't vary from one lagrange function to another, evaluate its inverse here
        matinv = Interpolator.approximation_matrix(centers, centers, par, order, type, solving=1)
        with open(output_fname, 'w', newline='') as out:
            csv_out = csv.writer(out)
            for ndx in range(np.shape(centers)[0]):
                # [approximation matrix]^{-1} * [0, 0, ... 0, 1, 0, ... 0] is equivalent to getting the kth column
                csv_out.writerow(matinv[:, ndx])

        return output_fname

    @staticmethod
    def largrange_evaluate(centers, sampledValue, evalPoints, lagrange_fname, exactValue, par, order=None, type=None):
        # read Lagrange coefficients from the .csv file
        lagrangefunctionCoeff = np.genfromtxt(lagrange_fname, delimiter=',')
        # initialize
        fVal = np.zeros(len(evalPoints))
        Interpolator.approximation_matrix(evalPoints, centers, par, order, type)
        for ndx in range(np.shape(centers)[0]):
            largrangeApproxValue = np.matmul(Interpolator.approxMatrix, lagrangefunctionCoeff[ndx,:])
            fVal = fVal + largrangeApproxValue*sampledValue[ndx]
        rmsError = Utilities.rms_error(fVal, exactValue)
        return fVal, rmsError


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
        voronoiVertices = Voronoi(points)
        voronoiDistanceMatrix = distance_matrix(voronoiVertices.vertices, points)
        selfDistanceMatrix = distance_matrix(points, points)
        np.fill_diagonal(selfDistanceMatrix, np.inf)
        fillDistance = np.max(voronoiDistanceMatrix.min(axis=0))/2.0
        separationRadius = np.min(selfDistanceMatrix)/2.0
        return fillDistance, separationRadius
