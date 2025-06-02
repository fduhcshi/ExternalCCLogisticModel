import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.linalg import inv
import sys
sys.path.append("../")

from generatedata_v2 import *
import Single as single
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,matthews_corrcoef

from datetime import datetime
from pathlib import Path

curr_dir = Path(__file__).parent
result_dir = curr_dir / 'result'

current_time = datetime.now().strftime("%m_%d_%H_%M_%S")


def calculate_y_prob(X,theta):
    alpha = theta[0]
    beta = theta[1:3]
    temp = np.exp(alpha+np.dot(X,beta))
    return temp/(1+temp)


rho  = 0

def get_Est(setting, kind):
    dim = setting['theta0'].shape[0] - 1
    t1 = time.time()
    data = GenerateData(**setting)
    X,y,Xy,test_data,gamma0 = data.generate(2,0)
    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((ones_column, X))
    t2 = time.time()
    
    if  "single" == kind:
        b = single.part1(np.array(X),y)
        theta_est, conv_idx = b.getEst()
        Ese_est = b.get_Ese(theta_est)
        pi_est = (expit(np.einsum('ij,j->i', X, theta_est))).mean() 
        return X, y, test_data,pi_est, theta_est, conv_idx, Ese_est 

def get_Est_sb(setting,num_iter,kind): # 
    Dict = {}
    Result= [] # record the \theta_est
    Conv = [] # convergence indicator
    Covariate = []
    Ese_est = []
    Pi_est = []
    Y = []
    i= 0 
    j = 0
    time_list_3 = []
    time_list_4 = []
    auc_list = []
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_score_list = []
    mcc_list = []

    while (i < num_iter):
        t3 = time.time()
        X, y,test_data, pi_est, theta_est, conv_idx, ese_est = get_Est(setting, kind)
        t4 = time.time()
        X_test = test_data[:,:2]
        Y_test = test_data[:,-1]
        y_pred_prob = calculate_y_prob(X_test,theta_est)
        y_pred = (y_pred_prob >= 0.5).astype(float)

        auc_ = roc_auc_score(Y_test, y_pred_prob)
        accuracy_ = accuracy_score(Y_test,y_pred)
        recall_ = recall_score(Y_test,y_pred)
        precision_ = precision_score(Y_test,y_pred)
        f1_score_ = f1_score(Y_test,y_pred)
        mcc_ = matthews_corrcoef(Y_test,y_pred)

        Conv.append(conv_idx)
        if (conv_idx):
            Result.append(theta_est)
            Covariate.append(X)
            Ese_est.append(ese_est)
            Y.append(y)
            Pi_est.append(pi_est)
            time_list_3.append(t3)
            time_list_4.append(t4)
            auc_list.append(auc_)
            accuracy_list.append(accuracy_)
            recall_list.append(recall_)
            precision_list.append(precision_)
            f1_score_list.append(f1_score_)
            mcc_list.append(mcc_)

            i += 1

        else:
            j += 1

    Dict[str(setting)] = Result
    Dict[str(setting)+"_Ese"] = Ese_est
    Dict[str(setting)+"_X"] = Covariate
    Dict[str(setting)+"Converge"] = Conv
    Dict[str(setting)+"_y"] = Y
    Dict[str(setting)+'_Pi'] = Pi_est
    Dict[str(setting)+'t3'] = time_list_3
    Dict[str(setting)+'t4'] = time_list_4
    Dict[str(setting)+'auc'] = auc_list
    Dict[str(setting)+'accuracy'] = accuracy_list
    Dict[str(setting)+'recall'] = recall_list
    Dict[str(setting)+'precision'] = precision_list
    Dict[str(setting)+'f1_score'] = f1_score_list
    Dict[str(setting)+'mcc'] = mcc_list

    return Dict


s2 = {'N' : 200000, 'theta0': np.array([-5,-2,2]),'n1': 800, 'n0' : 4000, 'P':0.0667}
s4 = {'N' : 200000, 'theta0': np.array([-4,2,2]), 'n1': 800, 'n0' : 4000, 'P':0.116}
s6 = {'N' : 200000, 'theta0': np.array([-5,-2,2]),'n1': 1500, 'n0' : 3000, 'P':0.0667}
s8 = {'N' : 200000, 'theta0': np.array([-4,2,2]),'n1': 1500, 'n0' : 3000, 'P':0.116}
s10 = {'N' : 200000, 'theta0': np.array([-5,-2,2]),'n1': 2000, 'n0' : 2000, 'P':0.0667}
s12 = {'N' : 200000, 'theta0': np.array([-4,2,2]),'n1': 2000, 'n0' : 2000, 'P':0.116}


num_iter = 1000

index = []
Temp = pd.DataFrame(columns = ['alpha_Bias','beta1_Bias','beta2_Bias','alpha_SE','beta1_SE','beta2_SE','alpha_Ese','beta1_Ese','beta2_Ese','alpha_CP','beta1_CP','beta2_CP']) #record the SE, Bias
dict_record = {}


for sets in tqdm([s2,s4,s6,s8,s10,s12]):
        index.append(str(sets))
        
        setting = sets.copy()
        setting.pop("P")

        Dict = get_Est_sb(setting, num_iter,'single') # get estimate
        dict_record[str(setting)] = Dict
        
        theta = np.array(Dict[str(setting)]) #theta_est
        n0 = setting['n0']
        n1 = setting['n1']
        ese = np.sqrt(np.array(Dict[str(setting)+'_Ese']))
        
        Pi_est = np.array(Dict[str(setting)+'_Pi'])

        time_list_3 = np.array(Dict[str(setting)+'t3'])
        time_list_4 = np.array(Dict[str(setting)+'t4'])
        auc_list = np.array(Dict[str(setting)+'auc'])
        accuracy_list = np.array(Dict[str(setting)+'accuracy'])
        recall_list = np.array(Dict[str(setting)+'recall'])
        precision_list = np.array(Dict[str(setting)+'precision'])
        f1_score_list = np.array(Dict[str(setting)+'f1_score'])
        mcc_list = np.array(Dict[str(setting)+'mcc'])

        temp = {}
        Ese_result = {}
        temp['P_Bias'] = Pi_est.mean() - sets['P']
        temp['alpha_Bias'] = theta.mean(axis = 0)[0] - setting['theta0'][0]
        temp['alpha_SE'] = np.sqrt(((theta[:,0] - theta[:,0].mean())**2).mean())
        
        theta_true = setting['theta0']
        temp['alpha_CP'] = (sum((theta_true >= (theta - 1.96*ese)) & (theta_true <= (theta + 1.96*ese)))/num_iter)[0]
        
        temp['alpha_Ese'] = ese[:,0].mean()
        
        temp['t3_mean'] = time_list_3.mean()
        temp['t4_mean'] = time_list_4.mean()
        temp['t3_sum'] = time_list_3.sum()
        temp['t4_sum'] = time_list_4.sum()
        temp['times'] = len(time_list_3)
        temp['auc'] = auc_list.mean()
        temp['accuracy'] = accuracy_list.mean()
        temp['recall'] = recall_list.mean()
        temp['precision'] = precision_list.mean()
        temp['f1'] = f1_score_list.mean()
        temp['mcc'] = mcc_list.mean()


        for j in range(1,setting['theta0'].shape[0]):
            temp['beta%d_Bias'%j] = theta.mean(axis = 0)[j] - setting['theta0'][j]
            temp['beta%d_SE'%j] = np.sqrt(((theta[:,j] - theta[:,j].mean())**2).mean())
            temp['beta%d_CP'%j] = (sum((theta_true >= (theta - 1.96*ese)) & (theta_true <= (theta + 1.96*ese)))/num_iter)[j]
            temp['beta%d_Ese'%j] = ese[:,j].mean()


        Temp = Temp.append(temp, ignore_index = True)
        
        
Temp.index = index
Temp.to_csv(result_dir/f'summary_single_cc_{current_time}.csv')

print(Temp)

