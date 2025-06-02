# -*- coding: utf-8 -*-
'''
'''
from scipy.optimize import minimize
from scipy.optimize import root ,fsolve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import approx_fprime
from statsmodels.tools.numdiff import approx_hess
np.seterr(divide='ignore',invalid='ignore')


class part:
    def __init__(self, X, y, W):
        self.X = X
        self.X_int = X[0:y.shape[0]]
        self.X_ext = X[y.shape[0]:]
        self.y = y
        self.n1 = int(sum(y))
        self.n0 = int(sum(1 - y))
        self.n = self.n1 + self.n0
        self.N = X.shape[0]-y.shape[0]
        self.v1 = self.n1/self.n
        self.X_0 = X[0:self.n0]
        self.W = W

    '''
    alpha_star = gamma + alpha
    here theta includes v, i.e. theta = (gamma, alpha_star, beta, u, v) 
    gamma = theta[0]
    alpha_star = theta[1]
    beta = theta[2:4]
    u = theta[4:6]
    v = theta[6:9]
    '''

    def exp_theta(self,X,theta):
        alpha_star = theta[1]
        beta = theta[2:4]
        temp = np.exp(np.clip(alpha_star + np.dot(X,beta),-700,700))
        return temp
    
    def exp_theta2(self,X,theta):
        gamma = theta[0]
        alpha = theta[1] - gamma
        beta = theta[2:4]
        temp = expit(gamma) * np.exp(np.clip(alpha + np.dot(X,beta),-700,700))
        return temp
    
    
    def H_x(self,theta):
        gamma = theta[0]
        beta = theta[2:4]
        u = theta[4:6]   
        H = np.zeros((self.X_int.shape[0], 3)) #dim H is n*3
        h_x = self.X_int
        H[:, 0] = self.exp_theta(self.X_int,theta) - 1
        H[:, 1] = (self.exp_theta2(self.X_int,theta) + expit(gamma)) * h_x[:,0] - u[0]
        H[:, 2] = (self.exp_theta2(self.X_int,theta) + expit(gamma)) * h_x[:,1] - u[1]
        return  H

    def H_x_gamma_der(self,theta):
        '''
        the derivative of H_x about gamma
        '''
        gamma = theta[0]
        H = np.zeros((self.X_int.shape[0], 3))
        h_x = self.X_int
        H[:, 1] = ((1-self.exp_theta(self.X_int,theta))*expit(gamma)) / (1 + np.exp(np.clip(gamma,-700,700))) * h_x[:,0]
        H[:, 2] = ((1-self.exp_theta(self.X_int,theta))*expit(gamma)) / (1 + np.exp(np.clip(gamma,-700,700))) * h_x[:,1]
        return H

    def H_x_alpha_der(self,theta):
        '''
        the derivative of H_x
        '''
        gamma = theta[0]
        H = np.zeros((self.X_int.shape[0], 3))
        h_x = self.X_int
        H[:, 0] = self.exp_theta(self.X_int,theta)
        H[:, 1] = (self.exp_theta2(self.X_int,theta)) * h_x[:,0]
        H[:, 2] = (self.exp_theta2(self.X_int,theta)) * h_x[:,1]
        return H

    def H_x_beta_der(self,theta):
        '''
        the derivative of H_x
        H[0] for beta1, H[1] for beta2
        '''
        H = np.zeros((2,self.X_int.shape[0], 3))
        H[0] = self.H_x_alpha_der(theta)*self.X_int[:,0].reshape([-1, 1]) 
        H[1] = self.H_x_alpha_der(theta)*self.X_int[:,1].reshape([-1, 1])
        return H

    def H_x_u_der(self):
        H = np.zeros((2,self.X_int.shape[0], 3))
        H[0,:,1] = -1.0 
        H[1,:,2] = -1.0
        return H


    def delta(self,theta):
        '''
        delta = e^(alpha* + X^T @ beta), rho = n1/n0, big_delta = 1 + rho*delta
        '''
        delta0 = self.exp_theta(self.X_0,theta)
        delta0 = delta0.reshape(-1,1)
        return delta0
    
    def dita(self,theta):
        delta0 = self.exp_theta(self.X_0,theta)
        dita = 1 + self.n1/self.n0*delta0
        dita = dita.reshape(-1,1)
        return dita
    
    def A15(self,theta):
        '''
        calculate A15 = E_0{partial H/ partial gamma}
        '''
        gamma = theta[0]
        H = np.zeros((self.X_0.shape[0], 3))
        h_x = self.X_0
        H[:, 1] = ((1-self.exp_theta(self.X_0,theta))*expit(gamma)) / (1 + np.exp(np.clip(gamma,-700,700))) * h_x[:,0]
        H[:, 2] = ((1-self.exp_theta(self.X_0,theta))*expit(gamma)) / (1 + np.exp(np.clip(gamma,-700,700))) * h_x[:,1]
        a15 = H.mean(axis=0)
        a15 = a15.reshape((1,3))
        return a15
    
    def A22(self,theta):
        rho = self.n1/self.n0
        temp = (self.delta(theta)/self.dita(theta)).mean()
        a22 = rho/(1+rho)*temp
        a22 = a22.reshape((1,1))
        return a22
    
    def A23(self,theta):
        '''1x2'''
        rho = self.n1/self.n0
        temp = (self.delta(theta)/self.dita(theta)*self.X_0).mean(axis=0)
        a23 = rho/(1+rho)*temp
        a23 = a23.reshape((1,2))
        return a23

    def A25(self,theta):
        '''1x3'''
        H_alpha_der = np.zeros((self.X_0.shape[0], 3))
        h_x = self.X_0
        H_alpha_der[:, 0] = self.exp_theta(self.X_0,theta)
        H_alpha_der[:, 1] = (self.exp_theta2(self.X_0,theta)) * h_x[:,0]
        H_alpha_der[:, 2] = (self.exp_theta2(self.X_0,theta)) * h_x[:,1]
        
        gamma = theta[0]
        u = theta[4:6]   
        H = np.zeros((self.X_0.shape[0], 3)) 
        H[:, 0] = self.exp_theta(self.X_0,theta) - 1
        H[:, 1] = (self.exp_theta2(self.X_0,theta) + expit(gamma)) * h_x[:,0] - u[0]
        H[:, 2] = (self.exp_theta2(self.X_0,theta) + expit(gamma)) * h_x[:,1] - u[1]
        H_dita = H/self.dita(theta)
        
        a25 = H_alpha_der.mean(axis=0) + H_dita.mean(axis=0)
        a25 = a25.reshape((1,3))
        return a25
    
    def A33(self,theta):
        rho = self.n1/self.n0
        XXT = self.X_0.T @ (self.delta(theta)/self.dita(theta)*self.X_0)
        a33 = rho/(1+rho)*XXT/self.n0
        return a33
    
    def A35(self,theta):
        '''
        2x3
        H[0] for beta1, H[1] for beta2
        '''
        H_alpha_der = np.zeros((self.X_0.shape[0], 3))
        h_x = self.X_0
        H_alpha_der[:, 0] = self.exp_theta(self.X_0,theta)
        H_alpha_der[:, 1] = (self.exp_theta2(self.X_0,theta)) * h_x[:,0]
        H_alpha_der[:, 2] = (self.exp_theta2(self.X_0,theta)) * h_x[:,1]
        
        Hbeta = np.zeros((2,self.X_0.shape[0], 3))
        Hbeta[0] = H_alpha_der*self.X_0[:,0].reshape([-1, 1])
        Hbeta[1] = H_alpha_der*self.X_0[:,1].reshape([-1, 1])
        E1 = Hbeta.mean(axis=1)

        h_x = self.X_0
        gamma = theta[0]
        u = theta[4:6]   
        H = np.zeros((self.X_0.shape[0], 3)) 
        H[:, 0] = self.exp_theta(self.X_0,theta) - 1
        H[:, 1] = (self.exp_theta2(self.X_0,theta) + expit(gamma)) * h_x[:,0] - u[0]
        H[:, 2] = (self.exp_theta2(self.X_0,theta) + expit(gamma)) * h_x[:,1] - u[1]
        
        E2 = (self.X_0.T @ (self.delta(theta)/self.dita(theta)*H))/self.n0
        a35 = E1 - self.n1/self.n0*E2
        return a35
    
    def A44(self):
        W = self.W
        return self.N * np.linalg.inv(W)/self.n
    
    def A45(self):
        '''2x3'''
        H = np.zeros((2, 3))
        H[0,1] = -1.0 
        H[1,2] = -1.0
        return H
    
    def A55(self,theta):
        '''3x3'''
        rho = self.n1/self.n0
        h_x = self.X_0
        gamma = theta[0]
        u = theta[4:6]   
        H = np.zeros((self.X_0.shape[0], 3)) #dim H is n*3
        H[:, 0] = self.exp_theta(self.X_0,theta) - 1
        H[:, 1] = (self.exp_theta2(self.X_0,theta) + expit(gamma)) * h_x[:,0] - u[0]
        H[:, 2] = (self.exp_theta2(self.X_0,theta) + expit(gamma)) * h_x[:,1] - u[1]
        
        a55 = (1+rho) * (H.T @ (H/self.dita(theta)))/self.n0
        return a55
    
    def calcu_Axx(self,theta):
        return self.A15(theta),self.A22(theta),self.A23(theta),self.A25(theta),self.A33(theta),self.A35(theta),self.A44(),self.A45(),self.A55(theta)
    
    def U(self,theta):
        A15,A22,A23,A25,A33,A35,A44,A45,A55 = self.calcu_Axx(theta)
        u11 = np.zeros((1,5))
        u23 = np.zeros((1,2))
        u33 = np.zeros((2,2))
        u41 = np.zeros((2,3))
        row1 = np.hstack((u11,A15))
        row2 = np.hstack((A22,A23,u23,A25))
        row3 = np.hstack((A23.T,A33,u33,A35))
        row4 = np.hstack((u41,A44,A45))
        U = np.vstack((row1,row2,row3,row4))
        return U
    
    def M(self,theta):
        A15,A22,A23,A25,A33,A35,A44,A45,A55 = self.calcu_Axx(theta)
        row1 = np.hstack((A22,A23,np.zeros((1,5))))
        row2 = np.hstack((A23.T,A33,np.zeros((2,5))))
        row3 = np.hstack((np.zeros((2,3)),A44,np.zeros((2,3))))
        row4 = np.hstack((np.zeros((3,5)),A55))
        M = np.vstack((row1,row2,row3,row4))
        return M
    
    def J(self,theta):
        J = self.U(theta) @ np.linalg.inv(self.M(theta)) @ self.U(theta).T
        return J
    
    def C(self):
        temp = np.zeros((6,6))
        temp[0:2,0:2] = 1
        rho = self.n1/self.n0
        temp = ((1+rho)**2 / rho) * temp
        return temp
    
    
    def asym_variance(self,theta):
        A15,A22,A23,A25,A33,A35,A44,A45,A55 = self.calcu_Axx(theta)
        c = np.vstack((A22,A23.T,0,0,A15.T+A25.T))
        rho = self.n1/self.n0
        V = np.array([[1, 0], [0, 1]])
        W_1 = np.linalg.inv(self.W)
        a = np.zeros((8, 8))
        b = -self.N/self.n * W_1 + self.N/self.n * (W_1 @ V @ W_1)
        rows, cols = b.shape
        a[3:3+rows,3:3+cols] = b
        temp1 = self.M(theta) + a - ((1+rho)**2/rho) * (c @ c.T)
        temp = np.linalg.inv(self.J(theta)) @ self.U(theta) @ np.linalg.inv(self.M(theta)) @ temp1 @ np.linalg.inv(self.M(theta)) @ self.U(theta).T @ np.linalg.inv(self.J(theta))
        return temp

    def ese_of_alpha(self,theta):
        matrix = self.asym_variance(theta)
        cov_a_g = matrix[0,1]
        v1 = matrix[0,0]
        v2 = matrix[1,1]
        result = np.sqrt((v1 + v2 - 2*cov_a_g)/self.n)
        return result
    
    def ese(self,theta):
        return np.sqrt(np.diagonal(self.asym_variance(theta))/self.n)

    def CP(self,theta,theta_true):
        temp = (theta_true >= (theta - 1.96*self.ese(theta))) & (theta_true <= (theta + 1.96*self.ese(theta)))
        temp = temp.astype(int)

        alpha_mle = theta[1] - theta[0]
        alpha_true = theta_true[1] - theta_true[0]
        temp_alpha = (alpha_true >= (alpha_mle - 1.96*self.ese_of_alpha(theta))) & (alpha_true <= (alpha_mle + 1.96*self.ese_of_alpha(theta)))
        temp_alpha = temp_alpha.astype(int)
        return np.append(temp,temp_alpha)
    
    def P_Yequals1(self,theta):
        temp = 1/(1+np.exp(theta[0]))
        return temp

    def loglikelihood(self,theta,v):
        '''
        return the negative loglikelihood 
        W a positive matrix
        '''
        gamma = theta[0]
        alpha_star = theta[1]
        beta = theta[2:4]
        u = theta[4:6]
        W = self.W
        sum1 = np.dot(self.y,(alpha_star + np.dot(self.X_int,beta)))
        sum2 = np.sum(
            np.log(
            (1+np.dot(self.H_x(theta),v))*(1+np.dot(self.H_x(theta),v) >= 1/self.n)
            + (1/(2*self.n - self.n**2*(1+np.dot(self.H_x(theta),v)))+1e-10)*(1+np.dot(self.H_x(theta),v) < 1/self.n)
            )
        )

        sum3 = self.N*np.einsum('i,ij,j->', np.mean(self.X_ext, axis=0) - u, np.linalg.inv(W), np.mean(self.X_ext, axis=0) - u)/2
        return -(sum1-sum2-sum3)

    def likelihood_der(self,theta,v): 
        '''
        the derivative of likelihood
        '''
        gamma = theta[0]
        alpha_star = theta[1]
        beta = theta[2:4]
        u = theta[4:6]
        W = self.W
        temp = np.array([0.0]*theta.shape[0])
        temp[0] = -np.sum((np.dot(self.H_x_gamma_der(theta),v))/(1+np.dot(self.H_x(theta),v)))
        temp[1] = np.sum(self.y)-np.sum((np.dot(self.H_x_alpha_der(theta),v))/(1+np.dot(self.H_x(theta),v)))
        temp[2] = np.dot(self.y,self.X_int[:,0])-np.sum((np.dot(self.H_x_beta_der(theta)[0],v))/(1+np.dot(self.H_x(theta),v)))
        temp[3] = np.dot(self.y,self.X_int[:,1])-np.sum((np.dot(self.H_x_beta_der(theta)[1],v))/(1+np.dot(self.H_x(theta),v)))
        temp[4] = -np.sum((np.dot(self.H_x_u_der()[0],v))/(1+np.dot(self.H_x(theta),v))) +self.N*np.dot(np.linalg.inv(W),(np.mean(self.X_ext,axis=0)-u))[0]
        temp[5] = -np.sum((np.dot(self.H_x_u_der()[1],v))/(1+np.dot(self.H_x(theta),v))) +self.N*np.dot(np.linalg.inv(W),(np.mean(self.X_ext,axis=0)-u))[1]
        return  -temp
    
    def likelihood_der2(self,theta,v):
        gamma = theta[0]
        alpha_star = theta[1]
        beta = theta[2:4]
        u = theta[4:6]
        W = self.W
        temp = np.array([0.0]*v.shape[0])
        temp[0] = -np.sum(np.dot(self.H_x(theta),np.array([1.0,0,0]))/(1+np.dot(self.H_x(theta),v)))
        temp[1] = -np.sum(np.dot(self.H_x(theta),np.array([0,1.0,0]))/(1+np.dot(self.H_x(theta),v)))
        temp[2] = -np.sum(np.dot(self.H_x(theta),np.array([0,0,1.0]))/(1+np.dot(self.H_x(theta),v)))
        return -temp
    
    
    def likelihood_der3(self,theta,v):
        def wrapped_loglikelihood(params):
            return self.loglikelihood(params,v)
        result = approx_fprime(theta, wrapped_loglikelihood, 1e-6)
        return result
    
    def hess_matrix(self,theta,v):
        def wrapped_loglikelihood_der(params):
            return self.likelihood_der3(params,v)
        result = approx_fprime(theta, wrapped_loglikelihood_der, epsilon=1e-6)
        return result
    
    def hess_matrix_2(self,theta,v):
        def wrapped_loglikelihood(params):
            return self.loglikelihood(params,v)
        result = approx_hess(theta, wrapped_loglikelihood, 1e-6)
        return result
    
    def hess_matrix_22(self,theta,v):
        params = np.hstack([theta,v])
        def wrapped_loglikelihood(params):
            theta = params[0:6]
            v = params[6:9]
            return self.loglikelihood(theta,v)
        result = approx_hess(params, wrapped_loglikelihood, 1e-6)
        return result

    def cons(self,theta,v):
        temp = np.array([0.0]*3)
        temp[0] = np.sum(np.dot(self.H_x(theta),np.array([1.0,0,0]))/(1+np.dot(self.H_x(theta),v)))
        temp[1] = np.sum(np.dot(self.H_x(theta),np.array([0,1.0,0]))/(1+np.dot(self.H_x(theta),v)))
        temp[2] = np.sum(np.dot(self.H_x(theta),np.array([0,0,1.0]))/(1+np.dot(self.H_x(theta),v)))
        return temp/self.n 
    
   
    def solve_v3_eq_set(self,v_temp,theta):
        gamma = theta[0]
        u = theta[4:6]
        h_x = self.X_int

        def f(v):
            eqs=[]
            eqs.append(self.n1+self.n*(-v[0]
                                       +v[1]*(expit(gamma) * np.sum(h_x[:,0] /(1+np.dot(self.H_x(theta),v)))- u[0])
                                       +v[2]*(expit(gamma) * np.sum(h_x[:,1] /(1+np.dot(self.H_x(theta),v)))- u[1])))
            eqs.append(np.dot(self.y,self.X_int[:,0])-np.sum((np.dot(self.H_x_beta_der(theta)[0],v))/(1+np.dot(self.H_x(theta),v))))
            eqs.append(np.dot(self.y,self.X_int[:,1])-np.sum((np.dot(self.H_x_beta_der(theta)[1],v))/(1+np.dot(self.H_x(theta),v)))) 
            return eqs
        
        rt = root(f,v_temp)
        return rt.x
    
        
    def estimate(self,theta,v):
        iter = 1
        max_iter = 100
        epsilon = 1e-6

        while True:
            theta0 = theta
            res_theta = minimize(lambda x: self.loglikelihood(x,v)
                                 ,theta
                                 ,jac = lambda x: self.likelihood_der3(x,v)
                                 , method='L-BFGS-B'
                                 )
            theta = res_theta.x
            v = self.solve_v3_eq_set(v,theta)
                        
            if np.max(np.abs(theta - theta0)) < epsilon: 
                break

            iter += 1
            if iter > max_iter:
                break

        return iter,res_theta,v
                                        
    def pi_hat(self,theta,v):
        return 1/(1+np.dot(self.H_x(theta),v))/self.n
    
    def V_hat(self,theta,v):
        weights = (self.exp_theta2(self.X_int,theta)+expit(theta[0])) / ((self.n)*(1+np.dot(self.H_x(theta),v)))
        w_diag = np.diag(weights)
        h_x = self.X_int
        bias = h_x - np.mean(self.X_ext,axis=0)
        W_hat = bias.T @ w_diag @ bias
        return W_hat
    
    def get_ese(self,theta,v):
        J_nv = -self.hess_matrix_22(theta,v)/self.n
        I_nv = -np.diag(np.diag(J_nv))
        invere_J = np.linalg.inv(J_nv)
        temp = (invere_J@I_nv@invere_J/self.n)
        return temp
    
    def get_ese2(self,theta,v):
        J_nv = -self.hess_matrix_2(theta,v)/self.n
        I_nv = -np.diag(np.diag(J_nv))
        invere_J = np.linalg.inv(J_nv)
        temp = (invere_J@I_nv@invere_J/self.n)
        return temp




