import numpy as np
import math
import itertools
from dataInterpolation.interpolator import Interpolator
from dataInterpolation.utilities import Utilities
import matplotlib.pyplot as plt
import csv

class LatticeLagrangeInterpolator:
    '''
        Finding local lagrange functions on a lattice. 
        We will find a single local Lagrange function and simply shift it from one point to another.
        The catch is we can only find functions on the interior of a domain. The domain consists of inner domain and external region.
        The width of the external region is the footpring radius of the local lagrange function.
        The domain should be square and 2D
        Implementation is limited to RBFs only.
        TODO: extend to polynomial reproduction. 
    '''
    def __init__(self, bounds, fillDistance, tuningParmeter=None, par=None, order=None, type=None) -> None:
        self.bounds = bounds
        self.fillDistance = fillDistance
        if tuningParmeter:
            self.tuningParameter = tuningParmeter
        else:
            self.tuningParameter = 2
        # number of points along x and y direction of the lattice
        self.xPoints = math.ceil(self.tuningParameter*abs(np.log(fillDistance)))
        self.yPoints = math.floor((bounds[1]-bounds[0])/self.fillDistance)
        self.footprintRadius = self.xPoints*self.fillDistance
        self.dim = 2
        if order:
            self.order = order
        else:
            self.order = 0
        if par:
            self.par = par
        else:
            self.par = []
        if type:
            self.type = type
        else: 
            self.type = "Thin plate splines"
            self.par = [1.0]
    
    def lagrange_function_0_0(self):
        '''
            Calculate the (0,0) local lagrange function
        '''
        # find the (0,0) point. It should be left bottom point + self.footprintRadius*[1,1] for 2D case
        originPoint = (self.bounds[0] + self.footprintRadius)*np.ones(self.dim)
        # find the points belonging to the lattice centered about (0,0) point
        centers = Utilities.generate_lattice(originPoint, self.footprintRadius, self.fillDistance)
        '''x = []
        y = []
        for ndx in centers:
            x.append(ndx[0])
            y.append(ndx[1])
        plt.scatter(x,y)
        plt.xlim(self.bounds[0],self.bounds[1])
        plt.ylim(self.bounds[0],self.bounds[1])
        plt.show()'''
        # find the RBF distance matrix for the set of centers within the lattice
        latticeEvaluator = Interpolator(centers, [], par=self.par, order=self.order, type=self.type)
        latticeEvaluator.approximation_matrix()
        rbf_matrix_DFT = Utilities.dft_2D(latticeEvaluator.rbf_matrix,self.xPoints)
        lagrageCoeff_DFT = pow(rbf_matrix_DFT,-1)
        print("DFT Coefficients calculated")
        lagrangeCoeff = Utilities.dft_inverse_2D(lagrageCoeff_DFT,self.xPoints)
        output_fname = 'dataDump\LatticeLagrange_' + self.type + '_'+ str(self.xPoints) + 'FootPrintPoints'+ '.csv'
        with open(output_fname, 'w', newline='') as out:
            csv_out = csv.writer(out)
            for ndx in range(np.shape(lagrangeCoeff)[0]):
                csv_out.writerow(lagrangeCoeff[ndx, :])
        return output_fname

    def eval_lagrange_function_0_0(self):