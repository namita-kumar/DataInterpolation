import numpy as np
from scipy.spatial import distance_matrix
from scipy.special import jv
import itertools
from math import factorial
from multiprocessing import Pool, Process, RawArray
import csv

from scipy.spatial.kdtree import KDTree
from dataInterpolation.utilities import Utilities

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
        # number of points
        self.n = np.shape(self.data)[0]
        # dimension
        self.dim = np.shape(self.data)[1]
        # number of polynomial terms
        if order:
            self.numberOfPolyTerms = int(Utilities.n_choose_r(self.dim+self.order, self.order))
        else:
            self.numberOfPolyTerms = 0

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
            # TODO: the thin plate spline equation is only for 2D. Must generalize
            self.rbf_matrix = np.power(self.rbf_matrix, 2*self.par[0])*np.log(self.rbf_matrix + 1e-10*np.ones(np.shape(self.rbf_matrix)))
        elif self.type == "Bessel":
            # bessel function of the first kind where par[0] is the order
            self.rbf_matrix = jv(self.par[0],self.rbf_matrix)

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
        ''' Finds the Global Lagrange functions about the data points as centers. The function coefficients are written to a csv file
            Returns:
                output_fname (string): Name of the .csv file with the Lagrange function coefficients
        '''
        # Proceed to calculate and save Lagrange functions to a .csv file
        # Name of the csv file has the type of RBF, order of polynomials, and number of centers
        output_fname = 'dataDump\Lagrange_' + self.type + 'Order' + str(self.order) + '_' + str(len(self.centers)) + 'Centers'+ '.csv'
        # Suppose we wish to find the lagrange function centered about k-th data point.
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
        ''' 
            Reads the lagrange funciton coefficients from a CSV file and quasi interpolates the function at evalPoints
        '''
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
    def lagrange_init_functions(centers, centersShape, lagrangefunctionCoeff, lagrangeCoeffShape, par, order, type, kdtree_setup, NoOfNeighbors):
        '''
            Initializes a global dictionary that can be accessed by all child processes of the lagrange interpolator
        '''
        # stores shared data in a global dictionary
        var_dict['centers'] = centers
        var_dict['centersShape'] = centersShape
        var_dict['lagrangefunctionCoeff'] = lagrangefunctionCoeff
        var_dict['lagrangeCoeffShape'] = lagrangeCoeffShape
        var_dict['par'] = par
        var_dict['order'] = order
        var_dict['type'] = type
        var_dict['kd_tree'] = kdtree_setup
        var_dict['NoOfNeighbors'] = NoOfNeighbors

    @staticmethod
    def lagrange_worker_functions(ndx):
        ''' TODO: assigning the coefficients for the polynomial is currently hard coded. Fix this.
            Finds the coeeficients for the lagrange function at center ndx. The points within the footprint of the Lagrange function are found by querying the KD tree.
        '''
        # generates the lagrange functions for each center
        centersMem_np = np.frombuffer(var_dict['centers']).reshape(var_dict['centersShape'])
        lagrangeMem_np = np.frombuffer(var_dict['lagrangefunctionCoeff']).reshape(var_dict['lagrangeCoeffShape'])
        lagrangeRHS = np.zeros(var_dict['centersShape'][0])
        lagrangeRHS[ndx] = 1
        #use KD tree to find the neighbors within 10h of ndx-th center
        kdtree_setup = var_dict['kd_tree']
        _ , neighbors_ndx = kdtree_setup.query(centersMem_np[ndx], var_dict['NoOfNeighbors'])
        # add the center to the list of neighbors
        # neighbors_ndx = np.concatenate(([ndx], neighbors_ndx), axis=0)
        neighbors = centersMem_np[neighbors_ndx]
        new_ndx = 0
        evaluator = Interpolator(neighbors, [], par=var_dict['par'], order=var_dict['order'], type=var_dict['type'])
        evaluator.approximation_matrix(solving=1)
        if var_dict['order']:
            lagrangefunctionCoeff = np.zeros(len(centersMem_np)+evaluator.numberOfPolyTerms)
            neighbors_ndx = np.concatenate((neighbors_ndx, [-3,-2,-1]), axis=0)
        else:
            lagrangefunctionCoeff = np.zeros(len(centersMem_np))
        lagrangefunctionCoeff[neighbors_ndx] = evaluator.distanceMatrixInverse[:, new_ndx]
        np.copyto(lagrangeMem_np[ndx,:], lagrangefunctionCoeff)

    def largrange_interpolant_parallel(self, NoOfNeighbors):
        ''' TODO: Fix it. Why is it so slow?
            Finds the coefficients for the lagrange functions at a given set of centers. Each lagrange funciton's footprint contains NoOfNeighbors points.
            The process is parallelized. Each worker function is in charge of finding a different center's Lagrange function. For each center, the NoOfNeighbors points are found using a KD tree.
            This function sets up the KD tree and stores the same in global dictionary. 
            The worker functions query this global KD tree. 
            The coefficients for all Lagrange functions is collected and then storred in a CSV file.

            Returns:
                output_fname (string): Name of the .csv file with the Lagrange function coefficients
        '''
        # set up KD tree
        kdtree_setup = KDTree(self.centers)
        # store centers in shared memory
        centersMem = RawArray('d', np.shape(self.centers)[0] * np.shape(self.centers)[1])
        centersMem_np = np.frombuffer(centersMem).reshape(np.shape(self.centers))
        np.copyto(centersMem_np, np.array(self.centers))

        # set up shared memory for Lagrange coefficients
        lagrangeCoeffShape = [self.n, self.n+self.numberOfPolyTerms]
        lagrangeCoeffMem = RawArray('d', lagrangeCoeffShape[0] * lagrangeCoeffShape[1])
        a_pool = Pool(processes=2, initializer=Interpolator.lagrange_init_functions,
                         initargs=(centersMem, np.shape(self.centers), lagrangeCoeffMem, lagrangeCoeffShape, self.par, self.order, self.type, kdtree_setup, NoOfNeighbors))
        
        a_pool.map(Interpolator.lagrange_worker_functions, range(np.shape(self.centers)[0]))

        # read pooled results
        lagrangeMem_np = np.frombuffer(lagrangeCoeffMem).reshape(lagrangeCoeffShape)
        output_fname = 'dataDump\Lagrange_' + self.type + 'Order' + str(self.order) + '_'+ str(len(self.centers)) + 'CentersMP'+ '.csv'
        with open(output_fname, 'w', newline='') as out:
             csv_out = csv.writer(out)
             for ndx in range(len(lagrangeMem_np)):
                 csv_out.writerow(lagrangeMem_np[ndx,:])
        return output_fname

