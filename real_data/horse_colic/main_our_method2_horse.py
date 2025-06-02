# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from datetime import datetime
current_time = datetime.now().strftime("%m_%d_%H_%M_%S")
import Single as single
from opt_of_real_data2_pdim import *
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,matthews_corrcoef



current_dir = Path(__file__).parent
csv_dir = current_dir / "horse_colic.csv"
results_dir = current_dir / 'result'
results_dir.mkdir(exist_ok=True)


'''
                              name     role        type demographic description units missing_values
0                          surgery  Feature     Integer        None        None  None            yes
1                              age  Feature     Integer         Age        None  None             no
2                  hospital_number  Feature     Integer        None        None  None             no
3               rectal_temperature  Feature  Continuous        None        None  None            yes
4                            pulse  Feature     Integer        None        None  None            yes
5                 respiratory_rate  Feature     Integer        None        None  None            yes
6       temperature_of_extremities  Feature     Integer        None        None  None            yes
7                 peripheral_pulse  Feature     Integer        None        None  None            yes
8                 mucous_membranes  Feature     Integer        None        None  None            yes
9            capillary_refill_time  Feature     Integer        None        None  None            yes
10                            pain  Feature     Integer        None        None  None            yes
11                     peristalsis  Feature     Integer        None        None  None            yes
12            abdominal_distension  Feature     Integer        None        None  None            yes
13                nasogastric_tube  Feature     Integer        None        None  None            yes
14              nasogastric_reflux  Feature     Integer        None        None  None            yes
15           nasogastric_reflux_ph  Feature  Continuous        None        None  None            yes
16        rectal_examination_feces  Feature     Integer        None        None  None            yes
17                         abdomen  Feature     Integer        None        None  None            yes
18              packed_cell_volume  Feature     Integer        None        None  None            yes
19                   total_protein  Feature     Integer        None        None  None            yes
20     abdominocentesis_appearance  Feature     Integer        None        None  None            yes
21  abdominocentesis_total_protein  Feature     Integer        None        None  None            yes
22                         outcome  Feature     Integer        None        None  None            yes
23                 surgical_lesion   Target     Integer        None        None  None             no
24                     lesion_site  Feature     Integer        None        None  None             no
25                     lesion_type  Feature     Integer        None        None  None             no
26                  lesion_subtype  Feature     Integer        None        None  None             no
27                         cp_data  Feature     Integer        None        None  None             no
'''


df = pd.read_csv(csv_dir)
df = df[['packed_cell_volume','abdominocentesis_appearance']].dropna(subset=['packed_cell_volume']).copy()


scaler = StandardScaler()
df['packed_cell_volume'] = scaler.fit_transform(df['packed_cell_volume'].values.reshape(-1,1))
df = df.rename(columns={'abdominocentesis_appearance': 'y'})


df_ext = df[df['y'].isnull()].reset_index(drop=True)

df_int = df[~df['y'].isnull()].reset_index(drop=True)
df_int.loc[:,'y'] = df_int.loc[:,'y'].map({1.0:0}).fillna(1).astype(int)

print('='*100)
print('P(Y=1) label_data:',df_int['y'].mean())
print('total data:',df.shape[0])
print('int_data:',df_int.shape[0],'case:',df_int[df_int.y == 1].shape[0],'control:',df_int[df_int.y == 0].shape[0])
print('ext_data:',df_ext.shape[0])
print('=====df_int\n',df_int.head(2))
print('====df_ext\n',df_ext.head(2))
print('='*100)


columns_to_use = ['packed_cell_volume']

## pretrain all trainnigdata to get theta_full
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(df_int[columns_to_use], df_int['y'])
print('params',model.get_params())
print("Intercept (alpha):", model.intercept_)
print("Coefficients (beta):", model.coef_.flatten())
print('='*50)
a = model.intercept_[0]
b1 = model.coef_.flatten()[0]



'''
---------------------------------------
Optimization terminated successfully.
         Current function value: 0.588101
         Iterations 5
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                  167
Model:                          Logit   Df Residuals:                      165
Method:                           MLE   Df Model:                            1
Date:                Thu, 19 Dec 2024   Pseudo R-squ.:                 0.04430
Time:                        22:23:16   Log-Likelihood:                -98.213
converged:                       True   LL-Null:                       -102.77
Covariance Type:            nonrobust   LLR p-value:                  0.002549
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.8250      0.174      4.745      0.000       0.484       1.166
x1             0.5790      0.204      2.844      0.004       0.180       0.978
==============================================================================
'''




def full_data(data): #data for our method
    df = data.copy()
    Xy0 = df[df.y == 0]
    Xy1 = df[df.y == 1]
    Xy = pd.concat([Xy0,Xy1],axis=0)
    Xy.insert(0,'x0',[1] * df.shape[0])
    X = np.array(Xy[['x0']+columns_to_use])
    y = np.array(Xy['y'])
    return X,y


def real_data(n0,n1): #data for our method
    
    ext_data = df_ext
    int_data = df_int

    df = int_data[columns_to_use+['y']]
    Xy0 = df[df.y == 0].sample(n0)
    Xy1 = df[df.y == 1].sample(n1)
    Xy = pd.concat([Xy0,Xy1],axis=0)

    test_data = df[~df.index.isin(Xy.index)].sample(70)
    test_data = np.array(test_data)

    y = np.array(Xy['y'])

    Xy_ext = ext_data[columns_to_use+['y']]
    Xy = pd.concat([Xy,Xy_ext],axis=0)

    X = np.array(Xy[columns_to_use]) 
    gamma0 = np.log((df.shape[0] - np.sum(df.y))/np.sum(df.y))

    return X,y,test_data,gamma0





def calculate_y_prob(X,theta):
    alpha = theta[1]-theta[0]
    beta = theta[2:2+X.shape[1]]
    temp = np.exp(alpha+np.dot(X,beta))
    return temp/(1+temp)

def calculate_y_prob_single(X,theta):
    alpha = theta[0]
    beta = theta[1:1+X.shape[1]]
    temp = np.exp(alpha+np.dot(X,beta))
    return temp/(1+temp)



#for full data inference
def full_data_inference(data):
    print('full data size:',data.shape)
    X_full,y_full = full_data(data)
    b_full = single.part1(X_full,y_full)
    theta_full,conv_idx = b_full.getEst()
    ese_full = b_full.get_Ese(theta_full)

    precision_score = None
    return theta_full,ese_full**0.5,precision_score


theta_full,ese_full,predict_score = full_data_inference(df_int)
theta_full = pd.DataFrame(np.array(theta_full))
ese_full = pd.DataFrame(np.array(ese_full)) 
df_full = pd.concat([theta_full.T,ese_full.T],axis=0)
df_full.columns = ['alpha','beta1']
df_full.index = ['theta','ese']
print('full data inference\n',df_full)
df_full.to_csv(results_dir/f'full_data_result.csv')



#for our method and singe cc inference
times = 12
n0 = 30
n1 = 30
W = np.array([[1]])
v_initial=np.array([n1/(n0+n1),0])
gamma_true000 = np.log(df_int[df_int.y == 0].shape[0]/df_int[df_int.y == 1].shape[0])
print('gamma_true000',gamma_true000)

#for our method
ese_all = []
ese_alpha = []
count = []
result = []
theta_all = []
P1 = []

auc_list = []
accuracy_list = []
recall_list = []
precision_list = []
f1_score_list = []
mcc_list = []

#for single case control
ese_list_single = []
theta_list_single = []
auc_list_single = []
accuracy_list_single = []
recall_list_single = []
precision_list_single = []
f1_score_list_single = []
mcc_list_single = []



for j in tqdm(range(times)):

    X,y,test_data,gamma0=real_data(n0,n1)
    X_test = test_data[:, :-1]
    Y_test = test_data[:,-1]

    X_single = X[:y.shape[0]]
    X_single = np.hstack((np.ones((X_single.shape[0], 1)), X_single))

    theta_true = np.array([gamma0, gamma0+a,0,0])
    v = v_initial
    theta = theta_true


    #for our method
    obj = part(X,y,W)
    iter,res_theta,res_v = obj.estimate(theta,v)

    y_pred_prob = calculate_y_prob(X_test,res_theta.x)
    y_pred = (y_pred_prob >= 0.5).astype(float)

    theta_all.append(res_theta.x)
    ese = obj.ese(res_theta.x)
    count.append(iter)
    result.append(res_theta.success)
    ese_all.append(ese)
    ese_alpha.append(obj.ese_of_alpha(res_theta.x))
    P1.append(obj.P_Yequals1(res_theta.x))


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




    #for single case control
    b = single.part1(np.array(X_single),y)
    theta_est, conv_idx = b.getEst()
    Ese_est = b.get_Ese(theta_est)
    theta_list_single.append(theta_est)
    ese_list_single.append(Ese_est)

    y_pred_prob_single = calculate_y_prob_single(X_test,theta_est)
    y_pred_single = (y_pred_prob_single >= 0.5).astype(float)

    auc_list_single.append(roc_auc_score(Y_test, y_pred_prob_single))
    accuracy_list_single.append(accuracy_score(Y_test,y_pred_single))
    recall_list_single.append(recall_score(Y_test,y_pred_single))
    precision_list_single.append(precision_score(Y_test,y_pred_single))
    f1_score_list_single.append(f1_score(Y_test,y_pred_single))
    mcc_list_single.append(matthews_corrcoef(Y_test,y_pred_single))


#for our method summary
df_count = pd.DataFrame(count,columns=['iterations'])
df1 = pd.DataFrame(theta_all,columns=['gamma','alpha*','beta1','miu1'])
df1['alpha'] = df1['alpha*'] - df1['gamma']
df_ese = pd.DataFrame(ese_all,columns=['ese_gamma','ese_alpha*','ese_beta1','ese_miu1'])
df_alpha_ese = pd.DataFrame({'ese_alpha': ese_alpha})
df_result = pd.DataFrame(result,columns=['_Result_'])
df_P1 = pd.DataFrame(P1,columns=['P1'])


df_auc = pd.DataFrame(auc_list,columns=['auc'])
df_accuracy = pd.DataFrame(accuracy_list,columns=['accuracy'])
df_recall = pd.DataFrame(recall_list,columns=['recall'])
df_precision = pd.DataFrame(precision_list,columns=['precision'])
df_f1 = pd.DataFrame(f1_score_list,columns=['f1_score'])
df_mcc = pd.DataFrame(mcc_list,columns=['mcc'])

df = pd.concat([df_result,df_count,df1,df_alpha_ese,df_ese,df_P1,df_auc,df_accuracy,df_recall,df_precision,df_f1,df_mcc],axis=1)
filtered_df = df[~((df['_Result_'] == False) 
                   |(abs(df['gamma'] - gamma_true000) >= 1)
                   |(abs(df['ese_alpha'])>10)
                   |(abs(df['beta1']-b1)>10)
                   ) ]
filtered_df.reset_index(drop=True, inplace=True)
print('filtered_df shape',filtered_df.shape)
df_head = filtered_df.head(min([100,len(filtered_df)]))
m = filtered_df.mean()
m.to_csv(results_dir/f'our_method_result.csv')
print('m',m)


#for single case control summary
ese_list_single = pd.DataFrame(ese_list_single,columns=['ese_alpha','ese_beta1'])**0.5
theta_list_single = pd.DataFrame(theta_list_single,columns=['alpha','beta1'])
auc_list_single = pd.DataFrame(auc_list_single,columns=['auc'])
accuracy_list_single = pd.DataFrame(accuracy_list_single,columns=['accuracy'])
recall_list_single = pd.DataFrame(recall_list_single,columns=['recall'])
precision_list_single = pd.DataFrame(precision_list_single,columns=['precision'])
f1_score_list_single = pd.DataFrame(f1_score_list_single,columns=['f1_score'])
mcc_list_single = pd.DataFrame(mcc_list_single,columns=['mcc'])

df_single = pd.concat([theta_list_single,ese_list_single,auc_list_single,accuracy_list_single,recall_list_single,precision_list_single,f1_score_list_single,mcc_list_single],axis=1)

df_single = df_single.head(min([100,df_single.shape[0]]))
print('df single shape',df_single.shape)
m_single = df_single.mean()
m_single.to_csv(results_dir/f'single_data_result.csv')
print('single',m_single)

