import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.linalg import inv,pinv


class part1():
    def __init__(self, X, y):
        """
        :param X: covariates :(N)*(k+1) 
        :param y:
        """
        self.X = X
        self.y = y
        self.n1 = sum(y)
        self.n0 = sum(1 - y)
        self.N = self.X.shape[0]
        self.k = X.shape[1]

    def phi(self, theta):
        """
        :param theta:
        :return: phi(theta;x) array
        """
        temp = np.einsum('i,ji->j', theta, self.X) 

        return expit(temp)

    def logLikelihood(self, theta):
        sum1 = np.einsum('i,i', self.y, np.log(self.phi(theta))) 
        sum2 = np.einsum('i,i', (1 - self.y), np.log((1 - self.phi(theta)))) 
        return -(sum1 + sum2)

    def deriveLikelihood(self, theta):
        temp = self.phi(theta)
        sum1 = np.einsum('i,ij,i->j', self.y, self.X, (1 - temp)) 
        sum2 = np.einsum('i,ij,i->j', (1 - self.y), self.X, temp)
        return sum2 - sum1

    def getEst(self):
        result = minimize(self.logLikelihood, np.array([0.1] * self.k), method='L-BFGS-B', jac=self.deriveLikelihood,

                          )  
        return np.array(result.x), result.success


    def get_Ese(self, theta):
        #
        temp = np.einsum('j,j,jk,jl->kl', self.phi(theta), (1 - self.phi(theta)), self.X, self.X)
        temp = pinv(temp)
        return np.diagonal(temp)