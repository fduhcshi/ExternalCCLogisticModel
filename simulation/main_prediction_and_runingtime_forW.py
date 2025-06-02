import pandas as pd
import numpy as np
from generatedata_v2 import *
import time
import csv
from datetime import datetime
from opt_v15_alpha_star import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,matthews_corrcoef
from pathlib import Path

curr_dir = Path(__file__).parent
result_dir = curr_dir / 'result'

current_time = datetime.now().strftime("%m_%d_%H_%M_%S")


def add_theta(theta_all,theta):
    if theta_all is None:    
        theta_all = theta
    else:
        theta_all = np.vstack((theta_all, theta))
    return theta_all


def calculate_y_prob(X,theta):
    alpha = theta[1]-theta[0]
    beta = theta[2:4]
    temp = np.exp(alpha+np.dot(X,beta))
    return temp/(1+temp)


s2 = {'a_b':np.array([-5,-2,2]),'n0':4000, 'n1': 800, 'p':0.067,'gamma_true':2.634}
s4 = {'a_b':np.array([-4,2,2]),'n0':4000, 'n1': 800, 'p':0.116,'gamma_true':2.031}
s6 = {'a_b':np.array([-5,-2,2]),'n0':3000, 'n1': 1500, 'p':0.067,'gamma_true':2.634}
s8 = {'a_b':np.array([-4,2,2]),'n0':3000, 'n1': 1500, 'p':0.116,'gamma_true':2.031}
s10 = {'a_b':np.array([-5,-2,2]),'n0':2000, 'n1': 2000, 'p':0.067,'gamma_true':2.634}
s12 = {'a_b':np.array([-4,2,2]),'n0':2000, 'n1': 2000, 'p':0.116,'gamma_true':2.031}

setting = [s2,s4,s6,s8,s10,s12]


summary = pd.DataFrame()

for sets in tqdm(setting):
    n0 = sets['n0']
    n1 = sets['n1']
    a = sets['a_b'][0]
    b1 = sets['a_b'][1]
    b2 = sets['a_b'][2]
    gamma_true = sets['gamma_true']
    p1 = sets['p']
    factor = 1

    gamma = []
    count = []
    theta_all = None
    v_all = None
    cp_all = None
    bias_all = None
    ese_all = None
    ese_alpha = []
    v1_all = []
    pbias = []
    result = []
    auc_list = []
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_score_list = []
    mcc_list = []
    time_list_1 = []
    time_list_2 = []
    time_list_3 = []
    time_list_4 = []
    time_list_5 = []



    W = np.array([[0.2,0],[0,2]])
    v_initial=np.array([n1/(n0+n1),0,0])
    theta_true = np.array([gamma_true, gamma_true+a,b1,b2,0,0])
    data = GenerateData([a,b1,b2], 200000, n0, n1, factor) 

    times = 1200
    for j in range(times):
        if j%100==0:
            print('repeat times:',j+1)

        t1 = time.time()
        X,y,Xy,test_data,gamma0 = data.generate(2,0) 
        gamma.append(gamma0)

        X_test = test_data[:,:2]
        Y_test = test_data[:,-1]

        theta_guess = np.array([gamma_true, a+gamma_true,b1,b2,0,0])+np.random.normal(0,0.05) 
    
        v = v_initial
        theta = theta_guess

        t2 = time.time()

        obj = part(X,y,W)

        iter,res_theta,res_v = obj.estimate(theta,v)

        t3 = time.time()
        
        y_pred_prob = calculate_y_prob(X_test,res_theta.x)
        y_pred = (y_pred_prob >= 0.5).astype(float)

        t4 = time.time()

        auc_ = roc_auc_score(Y_test, y_pred_prob)
        accuracy_ = accuracy_score(Y_test,y_pred)
        recall_ = recall_score(Y_test,y_pred)
        precision_ = precision_score(Y_test,y_pred)
        f1_score_ = f1_score(Y_test,y_pred)
        mcc_ = matthews_corrcoef(Y_test,y_pred)

        auc_list.append(auc_)
        accuracy_list.append(accuracy_)
        recall_list.append(recall_)
        precision_list.append(precision_)
        f1_score_list.append(f1_score_)
        mcc_list.append(mcc_)

        count.append(iter)
    
        result.append(res_theta.success)
        theta_all = add_theta(theta_all,res_theta.x) 
        v_all = add_theta(v_all,v)
        bias_all = add_theta(bias_all,res_theta.x - theta_true)
        cp_all = add_theta(cp_all,obj.CP(res_theta.x,theta_true))
        ese_all = add_theta(ese_all,obj.ese(res_theta.x))
        ese_alpha.append(obj.ese_of_alpha(res_theta.x))
        pbias.append(obj.P_Yequals1(res_theta.x)-obj.P_Yequals1(theta_true))
        
        t5 = time.time()
        time_list_1.append(t1)
        time_list_2.append(t2)
        time_list_3.append(t3)
        time_list_4.append(t4)
        time_list_5.append(t5)

    #汇总
    df0 = pd.DataFrame({'迭代次数':count})
    df1 = pd.DataFrame(theta_all,columns=['gamma','alpha*','beta1','beta2','miu1','miu2'])

    df_count = pd.DataFrame(count,columns=['iterations'])
    df_bias = pd.DataFrame(bias_all,columns=['bias_gamma','bias_alpha*','bias_beta1','bias_beta2','bias_miu1','bias_miu2'])
    df_cp = pd.DataFrame(cp_all,columns=['cp_gamma','cp_alpha*','cp_beta1','cp_beta2','cp_miu1','cp_miu2','cp_alpha'])
    df_ese = pd.DataFrame(ese_all,columns=['ese_gamma','ese_alpha*','ese_beta1','ese_beta2','ese_miu1','ese_miu2'])
    df_alpha_ese = pd.DataFrame({'ese of alpha': ese_alpha})
    df_p = pd.DataFrame(pbias,columns=['P_bias'])
    df_result = pd.DataFrame(result,columns=['_Result_'])

    df_auc = pd.DataFrame(auc_list,columns=['auc'])
    df_accuracy = pd.DataFrame(accuracy_list,columns=['accuracy'])
    df_recall = pd.DataFrame(recall_list,columns=['recall'])
    df_precision = pd.DataFrame(precision_list,columns=['precision'])
    df_f1 = pd.DataFrame(f1_score_list,columns=['f1_score'])
    df_mcc = pd.DataFrame(mcc_list,columns=['mcc'])
    
    df_t1 = pd.DataFrame(time_list_1,columns=['t1'])
    df_t2 = pd.DataFrame(time_list_2,columns=['t2'])
    df_t3 = pd.DataFrame(time_list_3,columns=['t3'])
    df_t4 = pd.DataFrame(time_list_4,columns=['t4'])
    df_t5 = pd.DataFrame(time_list_5,columns=['t5'])


    df = pd.concat([df_result,df_count,df1,df_bias,df_cp,df_alpha_ese,df_ese,df_p,df_auc,df_accuracy,df_recall,df_precision,df_f1,df_mcc,df_t1,df_t2,df_t3,df_t4,df_t5],axis=1)

    filtered_df = df[~((df['_Result_'] == False) |
                    (abs(df['gamma'] - gamma0) >= 0.5)|
                        (df['iterations'] == 101) 
                    ) ]
    filtered_df.reset_index(drop=True, inplace=True)


    df_head = filtered_df.head(min([1000,len(filtered_df)]))
    # Standard Error
    se_gamma = df_head['gamma'].std()
    se_alpha_star = df_head['alpha*'].std()
    s2 = (df_head['alpha*']-df_head['gamma']).std() #alpha 
    se_beta1 = df_head['beta1'].std()
    se_beta2 = df_head['beta2'].std()
    se_miu1 = df_head['miu1'].std()
    se_miu2 = df_head['miu2'].std()
    SE = [s2,se_gamma,se_alpha_star,se_beta1,se_beta2,se_miu1,se_miu2]

    SE = pd.Series(SE,index=['se_alpha','se_gamma','se_alpha_star','se_beta1','se_beta2','se_miu1','se_miu2'])

    m = df_head.mean()

    truevalue = np.array([gamma_true, gamma_true+a,a,b1,b2])
    tv = pd.Series(truevalue,index=['gamma_true','alphastar_true','alpha_true','beta1_true','beta2_true'])
    m = pd.concat([m,SE,tv])
    summary[f'{n0}_{n1}_{a}_{b1}_{b2}_{p1}'] = m
    df_head.to_csv(result_dir/f'1000times_W_factor{factor}_{a}_{b1}_{b2}_{n0}_{n1}_{current_time}.csv', index=False)

summary.to_csv(result_dir/f'summary_W_factor{factor}_{current_time}.csv')


print(summary)

