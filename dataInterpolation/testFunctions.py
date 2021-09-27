import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Halton


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
        if dim > 2:
            print("franke function works with only 2D cases")
            exit()
        f_dim = []
        for ndx in range(len(x)):
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
            ndxValue = 0.5+1*x[ndx][0]+1*x[ndx][1]+0.75*np.exp(-((9*x[ndx][0]-2)**2+(9*x[ndx][1]-2)**2)/4)
            + 0.75*np.exp(-((9*x[ndx][0]+1)**2/49+(9*x[ndx][1]+1)**2/10))
            + 0.5*np.exp(-((9*x[ndx][0]-7)**2+(9*x[ndx][1]-3)**2)/4)
            - 0.2*np.exp(-((9*x[ndx][0]-4)**2+(9*x[ndx][1]-7)**2))

            f_dim.append(ndxValue)
        return f_dim
    
    @staticmethod
    def wedland22_function(x, v):
        dim = np.shape(x)[1]
        f_dim = []
        for ndx in range(len(x)):
            ndxValue = np.sin(3*np.sum(x[ndx])) + Tools.wedland_2_2(0.25*np.linalg.norm(x[ndx])) + 2*Tools.wedland_2_2(0.5*np.linalg.norm(x[ndx]-v))-Tools.wedland_2_2(0.5*np.linalg.norm(x[ndx]+v))
            f_dim.append(ndxValue)
        return f_dim


class Tools:
    @staticmethod
    def wedland_2_2(r):
        saturated_r = (1-r) if ((1-r)>=0) else 0.0
        phi_2_2 = saturated_r**6 * (35*r**2 + 18*r + 3)
        return phi_2_2

    @staticmethod
    def wedland_2_3(r):
        saturated_r = (1-r) if ((1-r)>=0) else 0.0
        phi_2_3 = saturated_r**8 * (32*r**3 + 25*r**2 + 8*r + 1)
        return phi_2_3

def plot_3Dspace(x, fValue, title):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.array(x)[:, 0], np.array(x)[:, 1], fValue, 'bo')
    ax.set_xlabel("X1")
    ax.set_xlim([-1, 1])
    ax.set_ylabel("X2")
    ax.set_ylim([-1, 1])
    plt.title(title)


if __name__ == "__main__":
    n_samples = 289

    space = Space([(-1.0, 1.0), (-1.0, 1.0)])

    halton = Halton()
    x = halton.generate(space, n_samples)
    x = np.array(x)
    fValue = TestFunctions.cyclic_product(0, 1.0, x)
    plot_3Dspace(x, fValue, 'Cyclic Product')
    fValue = TestFunctions.franke_function( x)
    plot_3Dspace(x, fValue, 'Franke function')
    fValue = TestFunctions.franke_function_wBias(x)
    plot_3Dspace(x, fValue, 'Franke function with bias')
    fValue = TestFunctions.wedland22_function(x,[0.25,0.25])
    plot_3Dspace(x, fValue, 'Wedland')
    plt.show()
