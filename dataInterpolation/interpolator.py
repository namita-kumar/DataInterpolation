import numpy as np
from scipy.spatial import distance_matrix
import itertools
from math import factorial
from scipy.spatial import Voronoi
from multiprocessing import Pool, Process, RawArray
import csv
import pdb

from scipy.spatial.kdtree import KDTree

# A global dictionary for storing shared data for parallel computing
var_dict = {}

class Interpolator:
    def __init__(self, centers, sampledValue, data=None, par=None, order=None, type=None):
        ''' TODO: check compatibility between par and type
            centers (numpy array): [n,m] array. m n-dimensional points that are the centers for the radial basis functions
            sampledValue (numpy array): the result at the sampled points
            data (numpy array): [n,m] array where n is the dimension of the sample space and m is the number of samples. Is equal to centers if not provided.
            par (list): List of parameters corresponding to the selected radial basis function
            order (int): order of the polynomials if polynomial reproduction is desired
            type (str): Type of radail basis functions (Euclidean, Gaussian, Multiquadric, or Thin Plate Splines). Default is Euclidean
        '''
        self.centers = centers
        self.sampledValue = sampledValue
        if data:
            self.data = data
        else:
            self.data = centers
        self.order = order
        if par:
            self.par = par
        else:
            self.par = []
        if type:
            self.type = type

    def interpolate_evaluate(self, evalPoints, exactValue):
        ''' Approxiamtes a dataset (xData, sampledValue) and determines the error in approximation at evalPoints given the exact values

            Args:
                evalPoints (numpy array): list of points where the approximated function should be evaluated
                exactValue (numpy array): the actual values at the n_evalPoints

            Returns:
                functionCoeff (numpy array): list of coefficients for the selected radial basis functions
                approxValue (numpy array): the approximated function's values at evalPoints
                rmsError (float): error in approximation determined at evaPoints
        '''
        functionCoeff = self.interpolate()
        approxValue = self.evaluate(evalPoints, functionCoeff, self.centers, self.par, self.order, self.type)
        rmsError = Utilities.rms_error(approxValue, exactValue)
        return functionCoeff, approxValue, rmsError

    def interpolate(self):
        ''' Approxiamtes a dataset (xData, sampledValue)
            Returns:
                functionCoeff (numpy array): list of coefficients for the selected radial basis functions (and polynomials)
        '''
        # The equations are of the form [approximation matrix]*[coefficients]=[sampled values]
        # matinv is the inverse of the approximation matrix
        self.approximation_matrix(solving=1)
        RHS_value = self.sampledValue
        if self.order:
            # if polynomial reproduction is desired, add zeros to the RHS
            RHS_value = np.concatenate((RHS_value, np.zeros(self.numberOfPolyTerms)))

        functionCoeff = np.matmul(self.distanceMatrixInverse, RHS_value)
        return functionCoeff

    def approximation_matrix(self, solving=None):
        ''' Constructs the approximation matrix with includes the [RBF distance matrix, Polynomial matrix^T]
            If solving not None, constructs the inverse of the distance matrix for solving the linear equations.
            i.e. [RBF distance matrix, Polynomial matrix^T; Polynomial matrix, Zeros]
        '''
        self.distanceMatrixInverse = []
        if self.order:
            # if polynomial approximation is desired construct the polynomial matrix
            self.poly_matrix()
            # get the distance matrix with radial basis functions
            self.rbf_dist_matrix()
            # construct the approximation matrix
            self.approxMatrix = np.column_stack((self.rbf_matrix, np.transpose(self.pMatrix)))
            if solving:
                # if solving for the function coefficients, the approximation matrix should include additional equations
                # i.e. [coefficients]*[Polynomial matrix] = 0
                # a matrix of zeroes for padding
                z_matrix = np.zeros((self.numberOfPolyTerms, self.numberOfPolyTerms))
                distanceMatrix = np.concatenate((self.approxMatrix, np.column_stack((self.pMatrix, z_matrix))))
                self.distanceMatrixInverse = np.linalg.inv(distanceMatrix)
        else:
            # no polynomial reproximation
            self.rbf_dist_matrix()
            self.approxMatrix = self.rbf_matrix
            if solving:
                distanceMatrix = self.rbf_matrix
                self.distanceMatrixInverse = np.linalg.inv(distanceMatrix)

    def rbf_dist_matrix(self):
        ''' Generates the distance matrix given a radial basis function and its parameters.
            The radial basis functions are evaluated with data and centers.
            Based on self.type, different radial basis functions are selected.
        '''
        # default Euclidean distance matrix
        self.rbf_matrix = distance_matrix(self.data, self.centers)
        if self.type == "Gaussian":
            self.rbf_matrix = np.exp(-self.par[0]*np.power(self.rbf_matrix, 2))
        elif self.type == "Hardy multiquadric":
            self.rbf_matrix = np.power(self.rbf_matrix, 2) + self.par[0]**2*np.ones(np.shape(self.rbf_matrix))
            self.rbf_matrix = np.power(self.rbf_matrix, self.par[1]/2.0)
        elif self.type == "Thin plate splines":
            # added regularizing term to avoid taking log of zero
            self.rbf_matrix = np.power(self.rbf_matrix, 2*self.par[0])*np.log(self.rbf_matrix + 1e-10*np.ones(np.shape(self.rbf_matrix)))

    def poly_matrix(self):
        ''' Creates a vandermonde matrix pMatrix i.e. a matrix with the terms of a polynomial of n variables and m order.
            Given 2 data points with 2 variables [(x_1,y_1),(x_2,y_2)], this function Returns
            [[1, 1], [x_1, x_2], [y_1, y_2], [x_1^2, x_2^2], [y_1^2, y_2^2], [x_1*y_1, x_2*y_2]]
            The dimension of pMatrix is [number of data points, number of polynomial terms]
        '''
        n = np.shape(self.data)[1]
        mat = []
        for ndx in range(np.shape(self.data)[0]):
            # Suppose the number of variables is 2 (e.g. (x, y)) and the order is 2
            # the related polynomial terms are 1, x, y, x^2, y^2, xy
            # First: append 1. Note: this is order 0 term
            mat.extend([1])
            # Second: append terms in increasing order.
            # That is, append x, y first followed by x^2, y^2, xy
            for m in range(1, self.order+1):
                # Given the order of terms we are considering (i.e. m) find all corresponding combination of variables.
                # E.g. when m=2 and the variables are (x,y), itertools.product returns [(x,x),(y,y,),(x,y)]
                permute_x = itertools.product(self.data[ndx], repeat=m)
                # Take the product within the list [(x,x),(y,y),(x,y)] to get [x^2, y^2, xy] and then append to list
                mat.extend(list(map(np.prod, permute_x)))
        # Reshape the list to a 2D matrix. Total number of terms is given by (n+order)C(order)
        # The following gives the matrix [[1,x_1,y_1,x_1^2,y_1^2,x_1*y_1],[1,x_2,y_2,x_2^2,y_2^2,x_2*y_2], ...]
        self.numberOfPolyTerms = int(Utilities.n_choose_r(n+self.order, self.order))
        mat = np.reshape(mat, (np.shape(self.data)[0], self.numberOfPolyTerms))
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
        # creating an instance with the properties
        evaluator = Interpolator(centers, [], evalPoints, par, order, type)
        evaluator.approximation_matrix()
        approxValue = np.matmul(evaluator.approxMatrix, functionCoeff)
        return approxValue

    def largrange_interpolant(self):
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
        '''# Check if the mesh ratio is appropriate
        fillDistance, separationRadius = Utilities.fill_separation_distance(centers)
        meshRatio = fillDistance/separationRadius
        print("Mesh ratio=", meshRatio)
        if meshRatio>=2.5:
            print("ERROR: Mesh ratio is not suitable for accurate approximation. Try again.")
            exit()'''
        # Proceed to calculate and save Lagrange functions to a .csv file
        # Name of the csv file has the type of RBF, order of polynomials, and number of centers
        output_fname = 'dataDump\Lagrange_' + self.type + 'Order' + str(self.order) + '_' + str(len(self.centers)) + 'Centers'+ '.csv'
        # Suppose we wish to find the lagrange funciton centered about k-th data point.
        # The associated equations are [approximation matrix][coefficients] = [0, 0, ... 0, 1, 0, ... 0] where 1 is in the kth position
        # Since the approximation matrix doesn't vary from one lagrange function to another, evaluate its inverse here
        self.approximation_matrix(solving=1)
        with open(output_fname, 'w', newline='') as out:
            csv_out = csv.writer(out)
            for ndx in range(np.shape(self.centers)[0]):
                # [approximation matrix]^{-1} * [0, 0, ... 0, 1, 0, ... 0] is equivalent to getting the kth column
                csv_out.writerow(self.distanceMatrixInverse[:, ndx])

        return output_fname

    @staticmethod
    def largrange_evaluate(centers, sampledValue, evalPoints, lagrange_fname, exactValue, par, order=None, type=None):
        # read Lagrange coefficients from the .csv file
        lagrangefunctionCoeff = np.genfromtxt(lagrange_fname, delimiter=',')
        # initialize
        fVal = np.zeros(len(evalPoints))
        evaluator = Interpolator(centers, [], evalPoints, par, order, type)
        evaluator.approximation_matrix()
        for ndx in range(np.shape(centers)[0]):
            largrangeApproxValue = np.matmul(evaluator.approxMatrix, lagrangefunctionCoeff[ndx,:])
            fVal = fVal + largrangeApproxValue*sampledValue[ndx]
        rmsError = Utilities.rms_error(fVal, exactValue)

        return fVal, rmsError
    
    @staticmethod
    def lagrange_init_functions(centers, centersShape, par, order, type, kdtree_setup):
        # stores shared data in a global dictionary
        var_dict['centers'] = centers
        var_dict['centersShape'] = centersShape
        var_dict['par'] = par
        var_dict['order'] = order
        var_dict['type'] = type
        var_dict['kd_tree'] = kdtree_setup

    @staticmethod
    def lagrange_worker_functions(ndx):
        # generates the lagrange functions for each center
        centersMem_np = np.frombuffer(var_dict['centers']).reshape(var_dict['centersShape'])
        # print(var_dict['centersShape'])
        lagrangeRHS = np.zeros(var_dict['centersShape'][0])
        lagrangeRHS[ndx] = 1
        #use KD tree to find the neighbors within 10h of ndx-th center
        kdtree_setup = var_dict['kd_tree']
        neighbors_ndx = kdtree_setup.query_ball_point(centersMem_np[ndx], 2.0)
        neighbors = centersMem_np[neighbors_ndx]
        new_ndx = np.where(np.array(neighbors_ndx)==ndx)[0][0]
        evaluator = Interpolator(neighbors, [], par=var_dict['par'], order=var_dict['order'], type=var_dict['type'])
        evaluator.approximation_matrix(solving=1)
        lagrangefunctionCoeff = np.zeros(len(centersMem_np)+evaluator.numberOfPolyTerms)
        neighbors_ndx.extend([-3,-2,-1])
        lagrangefunctionCoeff[neighbors_ndx] = evaluator.distanceMatrixInverse[:, new_ndx]
        return lagrangefunctionCoeff

    def largrange_interpolant_parallel(self):
        kdtree_setup = KDTree(self.centers)
        centersMem = RawArray('d', np.shape(self.centers)[0] * np.shape(self.centers)[1])
        centersMem_np = np.frombuffer(centersMem).reshape(np.shape(self.centers))
        np.copyto(centersMem_np, np.array(self.centers))
        a_pool = Pool(processes=2, initializer=Interpolator.lagrange_init_functions,
                         initargs=(centersMem, np.shape(self.centers), self.par, self.order, self.type, kdtree_setup))
        
        lagrangefunctionCoeff = a_pool.map(Interpolator.lagrange_worker_functions, range(np.shape(self.centers)[0]))
        output_fname = 'dataDump\Lagrange_' + self.type + 'Order' + str(self.order) + '_'+ str(len(self.centers)) + 'CentersMP'+ '.csv'
        with open(output_fname, 'w', newline='') as out:
             csv_out = csv.writer(out)
             for ndx in range(len(lagrangefunctionCoeff)):
                 csv_out.writerow(lagrangefunctionCoeff[ndx])
        return output_fname




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
