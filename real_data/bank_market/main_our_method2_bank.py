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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,matthews_corrcoef


current_dir = Path(__file__).parent
csv_dir = current_dir / "bank_market.csv"
results_dir = current_dir / 'result'
results_dir.mkdir(exist_ok=True)


data = pd.read_csv(csv_dir)
data = data[['balance', 'duration','campaign','loan','poutcome','y']]
scaler = StandardScaler()
data[['balance', 'duration','campaign']] = scaler.fit_transform(data[['balance', 'duration','campaign']])
data['y'] = data['y'].map({'yes': 1, 'no': 0})
data['poutcome'] = data['poutcome'].map({'success': 1}).fillna(0).astype(int)
data['loan'] = data['loan'].map({'yes': 1}).fillna(0).astype(int)


print(data.shape[0],data[data.y == 1].shape[0],data[data.y == 0].shape[0])

print('P(Y=1) full data:',data['y'].mean(),f'//full data size{data.shape[0]}')
test_data = data.sample(1000)
test_data_index = test_data.index
data = data.drop(test_data_index)
test_data = np.array(test_data)
print('P(Y=1) after sample test data:',data['y'].mean(),f'//after test data size{data.shape[0]}')
X_test = test_data[:, :-1]
Y_test = test_data[:,-1]


columns_to_use = ['balance', 'duration','campaign','loan','poutcome']

## pretrain all trainnigdata to get theta_full
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(data[columns_to_use], data['y'])
print('params',model.get_params())
print("Intercept (alpha):", model.intercept_)
print("Coefficients (beta):", model.coef_.flatten())
print('='*50)
a = model.intercept_[0]
b1,b2,b3,b4,b5 = model.coef_[0].tolist()


'''
---------------------------------------
Optimization terminated successfully.
         Current function value: 0.271269
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                45211
Model:                          Logit   Df Residuals:                    45205
Method:                           MLE   Df Model:                            5
Date:                Tue, 17 Dec 2024   Pseudo R-squ.:                  0.2483
Time:                        01:37:38   Log-Likelihood:                -12264.
converged:                       True   LL-Null:                       -16315.
Covariance Type:            nonrobust   LLR p-value:                     0.000
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -2.4894      0.021   -117.272      0.000      -2.531      -2.448
x1             0.0954      0.014      6.967      0.000       0.069       0.122
x2             0.9597      0.015     64.167      0.000       0.930       0.989
x3            -0.3378      0.029    -11.552      0.000      -0.395      -0.280
x4            -0.6486      0.056    -11.514      0.000      -0.759      -0.538
x5             2.9239      0.060     48.618      0.000       2.806       3.042
==============================================================================
'''


def full_data(data): 
    df = data.copy()
    Xy0 = df[df.y == 0]
    Xy1 = df[df.y == 1]
    Xy = pd.concat([Xy0,Xy1],axis=0)
    Xy.insert(0,'x0',[1] * df.shape[0])
    X = np.array(Xy[['x0','balance', 'duration','campaign','loan','poutcome']])
    y = np.array(Xy['y'])

    return X,y


def real_data(data,n0,n1): #data for our method
    N = 10000
    
    ext_data = data.sample(N)
    int_data = data.drop(ext_data.index)

    df = int_data[columns_to_use+['y']]
    Xy0 = df[df.y == 0].sample(n0)
    Xy1 = df[df.y == 1].sample(n1)
    Xy = pd.concat([Xy0,Xy1],axis=0)

    y = np.array(Xy['y'])

    Xy_ext = ext_data[columns_to_use+['y']]
    Xy = pd.concat([Xy,Xy_ext],axis=0)

    X = np.array(Xy[columns_to_use])
    gamma0 = np.log((df.shape[0] - np.sum(df.y))/np.sum(df.y))

    return X,y,gamma0


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

    y_pred_prob_full = calculate_y_prob_single(X_test,theta_full)
    y_pred_full = (y_pred_prob_full >= 0.5).astype(float)

    auc_full = roc_auc_score(Y_test, y_pred_prob_full)
    accuracy_full = accuracy_score(Y_test,y_pred_full)
    recall_full = recall_score(Y_test,y_pred_full)
    precision_full = precision_score(Y_test,y_pred_full)
    f1_score_full = f1_score(Y_test,y_pred_full)
    mcc_full = matthews_corrcoef(Y_test,y_pred_full)

    predict_score = pd.DataFrame(np.array([auc_full,accuracy_full,recall_full,precision_full,f1_score_full,mcc_full]))
    
    return theta_full,ese_full**0.5,predict_score


theta_full,ese_full,predict_score = full_data_inference(data)
predict_score['score_name'] = ['auc','accuracy','recall','precision','f1_score','mcc']
theta_full = pd.DataFrame(np.array(theta_full))
ese_full = pd.DataFrame(np.array(ese_full)) 
df_full = pd.concat([theta_full.T,ese_full.T],axis=0)
df_full.columns = ['alpha','beta1','beta2','beta3','beta4','beta5']
df_full.index = ['theta','ese']
print('full data inference\n',df_full)
print('predict_score\n',predict_score)
df_full.to_csv(results_dir/f'bank_market_full_data_result.csv')
predict_score.to_csv(results_dir/f'bank_market_full_data_predict_score_result.csv')



#for our method and singe cc inference
times = 120
n0 = 2000
n1 = 2000
W = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
v_initial=np.array([n1/(n0+n1),0,0,0,0,0])
gamma_true000 = np.log(data[data.y == 0].shape[0]/data[data.y == 1].shape[0])


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

    X,y,gamma0=real_data(data,n0,n1)
    X_single = X[:y.shape[0]]
    X_single = np.hstack((np.ones((X_single.shape[0], 1)), X_single))


    theta_true = np.array([gamma0, gamma0+a,b1,b2,b3,b4,b5,0,0,0,0,0])
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
df1 = pd.DataFrame(theta_all,columns=['gamma','alpha*','beta1','beta2','beta3','beta4','beta5','miu1','miu2','miu3','miu4','miu5'])
df1['alpha'] = df1['alpha*'] - df1['gamma']
df_ese = pd.DataFrame(ese_all,columns=['ese_gamma','ese_alpha*','ese_beta1','ese_beta2','ese_beta3','ese_beta4','ese_beta5','ese_miu1','ese_miu2','ese_miu3','ese_miu4','ese_miu5'])
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
                   ) ]
filtered_df.reset_index(drop=True, inplace=True)
df_head = filtered_df.head(min([100,len(filtered_df)]))
m = filtered_df.mean()
m.to_csv(results_dir/f'bank_market_our_method_result.csv')
print('m',m)


#for single case control summary
ese_list_single = pd.DataFrame(ese_list_single,columns=['ese_alpha','ese_beta1','ese_beta2','ese_beta3','ese_beta4','ese_beta5'])**0.5
theta_list_single = pd.DataFrame(theta_list_single,columns=['alpha','beta1','beta2','beta3','beta4','beta5'])
auc_list_single = pd.DataFrame(auc_list_single,columns=['auc'])
accuracy_list_single = pd.DataFrame(accuracy_list_single,columns=['accuracy'])
recall_list_single = pd.DataFrame(recall_list_single,columns=['recall'])
precision_list_single = pd.DataFrame(precision_list_single,columns=['precision'])
f1_score_list_single = pd.DataFrame(f1_score_list_single,columns=['f1_score'])
mcc_list_single = pd.DataFrame(mcc_list_single,columns=['mcc'])

df_single = pd.concat([theta_list_single,ese_list_single,auc_list_single,accuracy_list_single,recall_list_single,precision_list_single,f1_score_list_single,mcc_list_single],axis=1)
m_single = df_single.mean()
m_single.to_csv(results_dir/f'bank_market_single_data_result.csv')
print('single',m_single)

