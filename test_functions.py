import numpy as np


class TestFunctions:
    @staticmethod
    def cyclic_product(x):
        # TODO: need to geenralize for space limits other than 1.0
        dim = np.shape(x)[1]
        f_dim = []
        for ndx in range(len(x)):
            ndxValue = 1.0
            for k in range(dim):
                ndxValue = ndxValue*x[ndx][k]*(1-x[ndx][k])
            ndxValue = 4**dim*ndxValue
            f_dim.append(ndxValue)
        return f_dim

    @staticmethod
    def franke_function(x):
        dim = np.shape(x)[1]
        if dim>2:
            print("franke function works with only 2D cases")
            exit()
        f_dim = []
        for ndx in range(len(x)):
            ndxValue = 0.0
            for k in range(dim):
                ndxValue = 0.75*np.exp(-((9*x[ndx][0]-2)**2+(9*x[ndx][1]-2)**2)/4)
                + 0.75*np.exp(-((9*x[ndx][0]+1)**2/49+(9*x[ndx][1]+1)**2/10))
                + 0.5*np.exp(-((9*x[ndx][0]-7)**2+(9*x[ndx][1]-3)**2)/4)
                - 0.2*np.exp(-((9*x[ndx][0]-4)**2+(9*x[ndx][1]-7)**2))
            f_dim.append(ndxValue)
        return f_dim

    @staticmethod
    def franke_function_wBias(x):
        dim = np.shape(x)[1]
        if dim>2:
            print("franke function works with only 2D cases")
            exit()
        f_dim = []
        for ndx in range(len(x)):
            ndxValue = 0.0
            for k in range(dim):
                ndxValue = 0.5+1*x[ndx][0]+1*x[ndx][1]+0.75*np.exp(-((9*x[ndx][0]-2)**2+(9*x[ndx][1]-2)**2)/4)
                + 0.75*np.exp(-((9*x[ndx][0]+1)**2/49+(9*x[ndx][1]+1)**2/10))
                + 0.5*np.exp(-((9*x[ndx][0]-7)**2+(9*x[ndx][1]-3)**2)/4)
                - 0.2*np.exp(-((9*x[ndx][0]-4)**2+(9*x[ndx][1]-7)**2))
            f_dim.append(ndxValue)
        return f_dim
