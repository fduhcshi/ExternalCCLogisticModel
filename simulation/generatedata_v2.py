import numpy as np
import pandas as pd
from scipy.special import expit
# from sklearn.utils import resample


class GenerateData:
    def __init__(self,theta0,N,n0,n1,factor=0):

        self.theta0 = theta0
        self.N = N
        self.n0 = n0
        self.n1 = n1
        self.factor = factor

    def phi(self,z):
        '''
        expitz = 1/1+e^-z
        '''
        return expit(z)

    def phi2(self,X):

        temp = np.einsum('i,ji->j', self.theta0, X)
        return self.phi(temp)

    def generate(self,dim,rho):

        X = pd.DataFrame({'x0':[1]*self.N})
        if dim ==1:
            for i in range(dim):
                temp = np.random.normal(size=self.N)
                X['x%d'%(i+1)] = temp
        else:

            X = pd.concat([X, pd.DataFrame(np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], self.N),columns=['x1', 'x2'])], axis=1)

        pi = self.phi2(np.array(X))
        vecF = np.vectorize(np.random.binomial)
        y = vecF(n=1, p=pi, size=1)
        data = pd.concat([X, pd.Series(y)], axis=1)
        data.columns = X.columns.tolist() + ['y']

        gamma0 = np.log(data[data.y == 0].shape[0]/data[data.y == 1].shape[0])
    
        data1 = pd.concat([data[data.y == 0].sample(self.n0),data[data.y == 1].sample(self.n1)])
        X1 = data1.iloc[:, :(dim + 1)]
        y1 = data1.iloc[:, -1]
        data1 = data1.iloc[:,1:]

        if self.factor >0:
            new_index = [_ for _ in data.index if _ not in data1.index]
            X2 = data.iloc[new_index, :(dim + 1)]
            X2 = X2.sample(int((self.n0 + self.n1) * self.factor))
            X1 = pd.concat([X1,X2]).reset_index(drop=True)

            test_index = [_ for _ in new_index if _ not in X2.index]
            test_data = data.iloc[test_index,1:].sample(int((self.n0 + self.n1)))

        if self.factor == 0:
            #增加test data
            test_index = [_ for _ in data.index if _ not in data1.index]
            test_data = data.iloc[test_index,1:].sample(int((self.n0 + self.n1)))

        X1 = X1.iloc[:, 1:(dim + 1)]
        return np.array(X1), np.array(y1),np.array(data1),np.array(test_data),gamma0




    
