# Last Updated 16th Sept
# Author asethi
# ML Assigment 1


#Importing packages
import researchpy as rp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Function to engineer features to be used in data modelling
# Features extracted from date
def feature_engineering(data):
    data['date']        = pd.to_datetime(data['date'])
    data["month"]       = data["date"].dt.month
    data["week_num"]    = data["date"].dt.week
    # Marking the flag if the launch day falls on the weekend
    data['weekday']     = data['date'].dt.weekday
    data["is_weekend"]  = data["date"].dt.weekday.apply(lambda x: 1 if x > 5 else 0)
    data["hour_of_day"] = data["date"].dt.hour
    data['time_of_day'] = data["hour_of_day"].apply(assign_time_of_day)
    data['time_of_day_encoded']=data.time_of_day.astype("category").cat.codes
    return data


# Function to assign time of the day, feature engineering continued
def assign_time_of_day(hour_val):
    if hour_val < 6:
        time = 'sleep_time'
    elif hour_val < 9:
        time = 'morning_time'
    elif hour_val <18:
        time = 'work_hours'
    else:
        time = 'night_time'
    return time

# Data Import from flatfile
energy_source_data=pd.read_csv('energydata_complete.csv')
# Invoking feature engineering function
energy_source_data_features=feature_engineering(energy_source_data)

# Original set of variables
XMaster=['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3',
       'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
       'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
       'Visibility', 'Tdewpoint', 'rv1', 'rv2', 'month',
       'is_weekend',  'time_of_day_encoded']

# Dependent Var
YMaster=['Appliances']

# Get correlation of all independent variables with dependent variables
dep_var_correl=energy_source_data_features[XMaster].apply(lambda x: x.corr(energy_source_data_features.Appliances)).to_frame()
#dep_var_correl
dep_var_correl.columns=['Correlation with Applainces']
dep_var_correl['Correlation with Applainces']=abs(dep_var_correl['Correlation with Applainces'])
dep_var_correl.sort_values(by=['Correlation with Applainces']).head(10)

# After getting correlation with dependent variable we calculate pairwise correlation

pairwise_correlation=rp.corr_pair(energy_source_data_features[['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3',
       'RH_3', 'T4', 'RH_4', 'T5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
       'RH_8', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
        'Tdewpoint', 'rv1', 'rv2', 'month',
       'is_weekend',  'time_of_day_encoded']])


pairwise_correlation=pd.DataFrame(pairwise_correlation)

pairwise_correlation['r value']=pairwise_correlation['r value']
pairwise_correlation.sort_values(by=['r value'], ascending=False).head(10)



# Intial Feature Selection after removing nine variables after correlation analysis
XMaster_Updated=['lights',  'RH_1', 'T2', 'RH_2', 'T3',
        'T4',  'RH_6', 'T7', 'RH_7', 'T8',
       'RH_8', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
        'Tdewpoint', 'rv2', 'month',
       'is_weekend',  'time_of_day_encoded']

# Scatter plots of dependent variable with selected independent variable
pp = sns.pairplot(data=energy_source_data_features,
                  y_vars=['Appliances'],
                  x_vars=['time_of_day_encoded', 'lights','month','is_weekend'])

# Distribution of values in Applainces column (Target Variable)
f = plt.figure(figsize=(8,3))
plt.xlabel('Appliance consumption in Wh')
plt.ylabel('Frequency')
sns.distplot(energy_source_data_features[['Appliances']] , bins=25 ) 
energy_source_data_features[['Appliances']].describe()


# Preparing dataset for linear regression
# Implementation steps in the function
# test and test partitioned into 77,5 and 22,5 split ratio
# the numerics transformed and scaled 
# an identity columns of ones is added to train and test dependent variables
# all datasets converted to numpy arrays

def data_prep_linear_regression(X,Y):
    scaler  = StandardScaler()
    X_train_out, X_test_out, Y_train_out, Y_test_out = train_test_split(X,Y,test_size=0.225,random_state=108)

    x_train = scaler.fit_transform(X_train_out)
    x_test  = scaler.fit_transform(X_test_out)

    m_train=x_train.shape[0]
    m_test =x_test.shape[0]
    
    c=x_train.shape[1]
    ones=np.ones((m_train,1))
    ones_alt = np.ones((m_test,1))
    X_train_mains = np.matrix(np.hstack((ones,x_train)))
    Y_train_mains = np.matrix(Y_train_out)
    X_test_mains = np.matrix(np.hstack((ones_alt,x_test)))
    Y_test_mains = np.matrix(Y_test_out)
    return X_train_mains,Y_train_mains,X_test_mains,Y_test_mains

# The function for linear regression,
# Implmentation steps:
# Theta which is the pararmeter array of weights of the regression coefficients 
# is randomly initialized, instead of random initialization we can also initialize it with numpy array of ones
# then the catch gradient descent is invoked within the function
def linear_regression_mains(X, y, alpha, num_iters,cost_threshold):
    theta = np.random.randn(X.shape[1])
    theta, cost, iter_runs = batch_gradient_descent_mains(theta,alpha,num_iters,X,y,cost_threshold)
    return theta, cost,iter_runs


# Function for batch gradient descent 
def batch_gradient_descent_mains(theta, alpha, num_iters,  X, Y,cost_threshold):
    #Initializing the array for cost calculations
    cost = np.ones(num_iters)
    
    for i in range(0,num_iters):
        #print(i)
          
        # Calculating hypothesis
        hypothesis_calculated = hypothesis_mains(theta, X)
        
        # Gradient is the differnce in y actual, calculated hypothesis, multiplied by 
        # learn rate and iteration count
        gradient = (alpha/X.shape[0]) * float(sum(hypothesis_calculated - Y))
        #updating theta 0 by subtracting gradient of this calculation
        
        theta[0] = theta[0] - gradient
        
        # for other thetas similar gradient calculation is done an the cost vals are simulataneously updated
        # by subtracting gradient
        for col in range(1,X.shape[1]):
            gradient_current = ((alpha/X.shape[0]) * float(sum(np.dot(X.T[col],(hypothesis_calculated-Y)))))
            theta[col] = theta[col] - gradient_current
        
        # cost of this iteration is calcuated simply by using the formula
        cost[i] = (1/X.shape[0]) * 0.5 * float(sum(np.square(hypothesis_calculated - Y)))
        
        # note : for i =0 we cannnot calculate the delta of costs, for the rest of i
        # we put a condition if the marginal change in cost is smaller than the given threshold exits out of 
        #iteration loop
        if(i>0):
            if(abs(cost[i]-cost[i-1])<cost_threshold):
                break
    #reshaping theta after final values are calculated
    theta = theta.reshape(1,X.shape[1])
    # total iterations ran are part ofthe output
    total_iterations_ran=i+1
    return theta, cost,total_iterations_ran


# Hypothesis calculation
# is similar to the dot product, it is estimated value of y bar
# for a given set of theta
def hypothesis_mains(theta, X):
    
    h_calc = np.ones((X.shape[0],1))
    
    theta = theta.reshape(1,X.shape[1])
    
    for i in range(0,X.shape[0]):
        h_calc[i] = float(np.dot(theta,X.T[:,i]))
        
    h_calc = h_calc.reshape(X.shape[0],1)
    return h_calc


# function to prepare data for linear regression evoked
X_train_mains,Y_train_mains,X_test_mains,Y_test_mains = data_prep_linear_regression(energy_source_data_features[XMaster_Updated],energy_source_data_features[YMaster])
# iter_max=1000
# cost_threshold=0.02
# optimal_weight, cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,0.15,iter_max, cost_threshold)


#prediction of test set and test set rmse calculation on the test set
def regression_prediction(optimal_weights,dep_vars,indep_vars):
    # we take optimal weight and dep vars as inputs and compute the dot product to calculate the predicted values
    predicted_values=dep_vars.dot(optimal_weights.transpose())
    #pred_values=predicted_values.to_numpy()
    #original_test_values=test_set['Appliances'].to_numpy()
    #Initializing RMSE
    rmse=0
    m=dep_vars.shape[0]
    
    # Differencing, squaring the differnec in prediction and actual and summing it all together,
    # before mean normalizing and calculating the root
    
    # rmse and predicted vars returned
    for i in range(m):
        rmse += (indep_vars[i] - predicted_values[i]) ** 2
    
    rmse = np.sqrt(rmse/m)
    return predicted_values,rmse

# predicted_values_train,rmse_train=regression_prediction(optimal_weight,X_train_mains,Y_train_mains)
# predicted_values_test,rmse_test=regression_prediction(optimal_weight,X_test_mains,Y_test_mains)

# print(rmse_train)
# print(rmse_test)
###
###
## Experiment 1 linear regression

# Max iterations are 1000, no threshold set for exp1
iter_max=1000
cost_threshold=0.00

# trace of cost array for all iterations intialized
cost_trace = np.array([])

# calculating cost trace for all learn rates
for learn_rate in [0.01,0.1,0.2,0.25]:
    
    optimal_weight,cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,learn_rate,iter_max, cost_threshold)
    cost_trace = np.append(cost_trace, cost)

# Plotting the convergence of cost fir train set
 
x = np.arange(999)

plt.plot(x, cost_trace[0:999])
plt.plot(x, cost_trace[1000:1999])
plt.plot(x, cost_trace[2000:2999])
plt.plot(x, cost_trace[3000:3999])

plt.legend(['learn_rate = 0.01 ', 'learn_rate = 0.1', 'learn_rate = 0.2', 'learn_rate = 0.25'], loc='upper right')

plt.show()

# Printing minimim cost of trainset for each learn rate
print ("Train Set Converging Cost for learning rate 0.01 ",cost_trace[999])
print ("Train Set Converging Cost for learning rate 0.1 ",cost_trace[1999])
print ("Train Set Converging Cost for learning rate 0.2 ",cost_trace[2999])
print ("Train Set Converging Cost for learning rate 0.25 ",cost_trace[3999])



###
### Invoking the same functions for experiment 1 for test data
iter_max=1000
cost_threshold=0.00
cost_trace = np.array([])
for learn_rate in [0.01,0.1,0.2,0.25]:
    
    optimal_weight,cost,iter_runs=linear_regression_mains(X_test_mains,Y_test_mains,learn_rate,iter_max, cost_threshold)
    cost_trace = np.append(cost_trace, cost)
    

# plotting convergence of cost with respect to 1000 iterations for different valued of learn rate on train set
x = np.arange(999)

plt.plot(x, cost_trace[0:999])
plt.plot(x, cost_trace[1000:1999])
plt.plot(x, cost_trace[2000:2999])
plt.plot(x, cost_trace[3000:3999])

plt.legend(['learn_rate = 0.01 ', 'learn_rate = 0.1', 'learn_rate = 0.2', 'learn_rate = 0.25'], loc='upper right')

plt.show()

# getting lowerst cost for each alpha
print ("Test Set Converging Cost for learning rate 0.01 ",cost_trace[999])
print ("Test Set Converging Cost for learning rate 0.1 ",cost_trace[1999])
print ("Test Set Converging Cost for learning rate 0.2 ",cost_trace[2999])
print ("Test Set Converging Cost for learning rate 0.25 ",cost_trace[3999])


# Running Gradient Descent and calculating train and test set error for learn rate 0.01 
optimal_weight_alpha1,cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,0.01,iter_max, cost_threshold)

train_pred_values,rmse_train_alpha1 = regression_prediction(optimal_weight_alpha1,X_train_mains,Y_train_mains)
test_pred_values,rmse_test_alpha1 = regression_prediction(optimal_weight_alpha1,X_test_mains,Y_test_mains)
print ("RMSE train for learn rate 0.01 == ",rmse_train_alpha1)
print ("RMSE test for learn rate 0.01 == ",rmse_test_alpha1)


# Running Gradient Descent and calculating train and test set error for learn rate 0.1
optimal_weight_alpha2,cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,0.1,iter_max, cost_threshold)

train_pred_values,rmse_train_alpha2 = regression_prediction(optimal_weight_alpha2,X_train_mains,Y_train_mains)
test_pred_values,rmse_test_alpha2 = regression_prediction(optimal_weight_alpha2,X_test_mains,Y_test_mains)
print ("RMSE train for learn rate 0.1 == ",rmse_train_alpha2)
print ("RMSE test for learn rate 0.1 == ",rmse_test_alpha2)



# Running Gradient Descent and calculating train and test set error for learn rate 0.2
optimal_weight_alpha3,cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,0.2,iter_max, cost_threshold)

train_pred_values,rmse_train_alpha3 = regression_prediction(optimal_weight_alpha3,X_train_mains,Y_train_mains)
test_pred_values,rmse_test_alpha3 = regression_prediction(optimal_weight_alpha3,X_test_mains,Y_test_mains)
print ("RMSE train for learn rate 0.2 == ",rmse_train_alpha3)
print ("RMSE test for learn rate 0.2 == ",rmse_test_alpha3)



# Running Gradient Descent and calculating train and test set error for learn rate 0.25
optimal_weight_alpha4,cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,0.25,iter_max, cost_threshold)

train_pred_values,rmse_train_alpha4 = regression_prediction(optimal_weight_alpha4,X_train_mains,Y_train_mains)
test_pred_values,rmse_test_alpha4 = regression_prediction(optimal_weight_alpha4,X_test_mains,Y_test_mains)
print ("RMSE train for learn rate 0.25 == ",rmse_train_alpha4)
print ("RMSE test for learn rate 0.25 == ",rmse_test_alpha4)


############# END OF experiment 1 for linear regression

# We increse the maximum possible iterations and fixate the learn rate to the optimum value

iter_max=1200
learn_rate = 0.25

# Getting output of regression for 4 values of threshold on train set
optimal_weight_t1,cost_train_t1,iter_runs_t1_train=linear_regression_mains(X_train_mains,Y_train_mains,learn_rate,iter_max, 0.5)
print ("For 0.5 threshold train set runs for == ",iter_runs_t1_train)
optimal_weight_t2,cost_train_t2,iter_runs_t2_train=linear_regression_mains(X_train_mains,Y_train_mains,learn_rate,iter_max, 0.1)
print ("For 0.1 threshold train set runs for == ",iter_runs_t2_train)
optimal_weight_t3,cost_train_t3,iter_runs_t3_train=linear_regression_mains(X_train_mains,Y_train_mains,learn_rate,iter_max, 0.05)
print ("For 0.05 threshold train set runs for == ",iter_runs_t3_train)
optimal_weight_t4,cost_train_t4,iter_runs_t4_train=linear_regression_mains(X_train_mains,Y_train_mains,learn_rate,iter_max, 0.02)
print ("For 0.02 threshold train set runs for == ",iter_runs_t4_train)


# Getting output of regression for 4 values of threshold on test set
optimal_weight_t1_test,cost_test_t1,iter_runs_t1_test=linear_regression_mains(X_test_mains,Y_test_mains,learn_rate,iter_max, 0.5)
print ("For 0.5 threshold test set runs for == ",iter_runs_t1_test)
optimal_weight_t2_test,cost_test_t2,iter_runs_t2_test=linear_regression_mains(X_test_mains,Y_test_mains,learn_rate,iter_max, 0.1)
print ("For 0.1 threshold test set runs for == ",iter_runs_t2_test)

optimal_weight_t3_test,cost_test_t3,iter_runs_t3_test=linear_regression_mains(X_test_mains,Y_test_mains,learn_rate,iter_max, 0.05)
print ("For 0.05 threshold test set runs for == ",iter_runs_t3_test)
optimal_weight_t4_test,cost_test_t4,iter_runs_t4_test=linear_regression_mains(X_test_mains,Y_test_mains,learn_rate,iter_max, 0.02)
print ("For 0.02 threshold test set runs for == ",iter_runs_t4_test)

train_pred_values_t1,rmse_train_t1 = regression_prediction(optimal_weight_t1,X_train_mains,Y_train_mains)
print ("RMSE train for threshold 0.5 == ",rmse_train_t1)

train_pred_values_t2,rmse_train_t2 = regression_prediction(optimal_weight_t2,X_train_mains,Y_train_mains)
print ("RMSE train for threshold 0.1 == ",rmse_train_t2)

train_pred_values_t3,rmse_train_t3 = regression_prediction(optimal_weight_t3,X_train_mains,Y_train_mains)
print ("RMSE train for threshold 0.05 == ",rmse_train_t3)

train_pred_values_t4,rmse_train_t4 = regression_prediction(optimal_weight_t4,X_train_mains,Y_train_mains)
print ("RMSE train for threshold 0.02 == ",rmse_train_t4)

test_pred_values_t1,rmse_test_t1 = regression_prediction(optimal_weight_t1,X_test_mains,Y_test_mains)
print ("RMSE test for threshold 0.5 == ",rmse_test_t1)

test_pred_values_t2,rmse_test_t2 = regression_prediction(optimal_weight_t2,X_test_mains,Y_test_mains)
print ("RMSE test for threshold 0.1 == ",rmse_test_t2)

test_pred_values_t3,rmse_test_t3 = regression_prediction(optimal_weight_t3,X_test_mains,Y_test_mains)
print ("RMSE test for threshold 0.05 == ",rmse_test_t3)

test_pred_values_t4,rmse_test_t4 = regression_prediction(optimal_weight_t4,X_test_mains,Y_train_mains)
print ("RMSE test for threshold 0.02 == ",rmse_test_t4)

# Consolidating cost trace in dataframe and plotting for differne t thresolds (train set)
train_cost_df_plot = pd.concat([pd.DataFrame(cost_train_t1),pd.DataFrame(cost_train_t2),pd.DataFrame(cost_train_t3),pd.DataFrame(cost_train_t4)], ignore_index=True, axis=1)
train_cost_df_plot.columns=['Threshold - 0.5','Threshold - 0.1','Threshold - 0.05','Threshold - 0.02']
train_cost_df_plot.shape

x = np.arange(1200)

plt.plot(x, train_cost_df_plot['Threshold - 0.5'])
plt.plot(x, train_cost_df_plot['Threshold - 0.1'])
plt.plot(x, train_cost_df_plot['Threshold - 0.05'])
plt.plot(x, train_cost_df_plot['Threshold - 0.02'])

#plt.plot(x, cost_trace[1000:1999])
#plt.plot(x, cost_trace[2000:2999])
#plt.plot(x, cost_trace[3000:3999])

plt.legend(['train-set-threshold = 0.5 ', 'train-set-threshold = 0.1', 'train-set-threshold = 0.05', 'train-set-threshold = 0.02'], loc='upper right')

plt.show()
# Getting values for lowerst converging cost for all threshold, train set
print ("Train Set Converging Cost for thrshold  0.5 ",cost_train_t1[iter_runs_t1_train-1])
print ("Train Set Converging Cost for thrshold  0.1 ",cost_train_t2[iter_runs_t2_train-1])
print ("Train Set Converging Cost for thrshold  0.05 ",cost_train_t3[iter_runs_t3_train-1])
print ("Train Set Converging Cost for thrshold  0.02 ",cost_train_t4[iter_runs_t4_train-1])


# Consolidating cost trace in dataframe and plotting for differne t thresolds (test set)

test_cost_df_plot = pd.concat([pd.DataFrame(cost_test_t1),pd.DataFrame(cost_test_t2),pd.DataFrame(cost_test_t3),pd.DataFrame(cost_test_t4)], ignore_index=True, axis=1)
test_cost_df_plot.columns=['Threshold - 0.5','Threshold - 0.1','Threshold - 0.05','Threshold - 0.02']
test_cost_df_plot.shape

x = np.arange(1200)

plt.plot(x, test_cost_df_plot['Threshold - 0.5'])
plt.plot(x, test_cost_df_plot['Threshold - 0.1'])
plt.plot(x, test_cost_df_plot['Threshold - 0.05'])
plt.plot(x, test_cost_df_plot['Threshold - 0.02'])


plt.legend(['test_set-threshold = 0.5 ', 'test_set-threshold = 0.1', 'test_set-threshold = 0.05', 'test_set-threshold = 0.02'], loc='upper right')

plt.show()
# Getting values for lowerst converging cost for all threshold, train set
print ("Test Set Converging Cost for thrshold  0.5 ",cost_test_t1[iter_runs_t1_test-1])
print ("Test Set Converging Cost for thrshold  0.1 ",cost_test_t2[iter_runs_t2_test-1])
print ("Test Set Converging Cost for thrshold  0.05 ",cost_test_t3[iter_runs_t3_test-1])
print ("Test Set Converging Cost for thrshold  0.02 ",cost_test_t4[iter_runs_t4_test-1])


# Most optimal model with 21 variables, invoking linear regressin
optimal_weight_varset1,cost_train,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,0.25,iter_max, 0.05)
# train and test set prediction for the above model
train_pred_values_varset1,rmse_train_varset1 = regression_prediction(optimal_weight_varset1,X_train_mains,Y_train_mains)
test_pred_values_varset1,rmse_test_varset1 = regression_prediction(optimal_weight_varset1,X_test_mains,Y_test_mains)

# getting required parameters
print ("Optimal weights for optimal model with 21 vars :  ",optimal_weight_varset1)
print ("Optimal learning for optimal model with 21 vars :  ",0.25)
print ("Optimal threshold for optimal model with 21 vars :  ",0.05)
print ("RMSE train for optimal model with 21 vars :  ",rmse_train_varset1)
print ("RMSE test for optimal model with 21 vars :  ",rmse_test_varset1)
print ("Total Iterations :  ",iter_runs)


# For Randomly chosen features experiment 3
# selecting 10 features randomly
energy_source_data_features_sampled_random=energy_source_data_features[XMaster_Updated].sample(10,axis=1)
print(energy_source_data_features_sampled_random.columns)
# preparing data for input dataset of 10 features
X_train_mains,Y_train_mains,X_test_mains,Y_test_mains = data_prep_linear_regression(energy_source_data_features_sampled_random,energy_source_data_features[YMaster])

# setting parametrts for model
iter_max=1000
learn_rate=0.15
cost_threshold=0

# running regression for experiment 3
optimal_weight_random, cost_random,iter_runs_random=linear_regression_mains(X_train_mains,Y_train_mains,learn_rate,iter_max, cost_threshold)

# train and test set prediction for experiment 3
predicted_values_train,rmse_train=regression_prediction(optimal_weight_random,X_train_mains,Y_train_mains)
predicted_values_test,rmse_test=regression_prediction(optimal_weight_random,X_test_mains,Y_test_mains)
print(rmse_train)
print(rmse_test)
print(cost_random)
print(optimal_weight_random)
x = np.arange(1000)

# plotting cost convergence for one value of learn rate for exp3 
plt.plot(x, cost_random)
plt.legend(['learn_rate = 0.25 '], loc='upper right')

plt.show()


######################################
# Experiment 4

# seperating vars that are to be one hot encoded
var_list = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3',
       'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
       'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
       'Visibility', 'Tdewpoint', 'rv1', 'rv2']

var_ohe_list = ['month','is_weekend', 'time_of_day']

# ohe
dataset_mains = pd.concat([energy_source_data_features[var_list],pd.get_dummies(energy_source_data_features[var_ohe_list])],axis=1)

# train and test data after onehot encoding
X = dataset_mains
Y = energy_source_data_features[YMaster]

# correlation calculated same as before
dep_var_correl=dataset_mains.apply(lambda x: x.corr(energy_source_data_features.Appliances)).to_frame()
#dep_var_correl
dep_var_correl.columns=['Correlation with Applainces']
dep_var_correl['Correlation with Applainces']=abs(dep_var_correl['Correlation with Applainces'])
dep_var_correl.sort_values(by='Correlation with Applainces', ascending=False)
#important_features= ['RH_8','Tdewpoint', 'RH_6', 'RH_5','rv1','T6','RH_7','rv2','T_out','RH_9']

# top 10 important features after sorting
important_features=['time_of_day_sleep_time','time_of_day_work_hours','lights','RH_out','T2','T6','T_out','RH_8','time_of_day_night_time','Windspeed']

# function to prepare data for linear regression
X_train_mains,Y_train_mains,X_test_mains,Y_test_mains = data_prep_linear_regression(X[important_features],Y)

# run regression
optimal_weight_pc10, cost,iter_runs=linear_regression_mains(X_train_mains,Y_train_mains,0.25,1000, 0)

# prediction on traina and test set to get error metrics
predicted_values_train,rmse_train=regression_prediction(optimal_weight_pc10,X_train_mains,Y_train_mains)
predicted_values_test,rmse_test=regression_prediction(optimal_weight_pc10,X_test_mains,Y_test_mains)
print(rmse_train)
print(rmse_test)
print(cost)

# running for 4 thresholds , model with optimal features
i=1200
optimal_weight_thesh1, cost_t1,iter_runs_t1=linear_regression_mains(X_train_mains,Y_train_mains,0.25,i,0.5)
print('model run for w1')
optimal_weight_thesh2, cost_t2,iter_runs_t2=linear_regression_mains(X_train_mains,Y_train_mains,0.25,i,0.1)
print('model run for w2')
optimal_weight_thesh3, cost_t3,iter_runs_t3=linear_regression_mains(X_train_mains,Y_train_mains,0.25,i,0.05)
print('model run for w3')
optimal_weight_thesh4, cost_t4,iter_runs_t4=linear_regression_mains(X_train_mains,Y_train_mains,0.25,i,0.02)
print('model run for w4')
optimal_weight_thesh5, cost_t5,iter_runs_t5=linear_regression_mains(X_train_mains,Y_train_mains,0.25,i,0.005)

# test set prediction for above regressions
#predicted_values_train,rmse_train=regression_prediction(optimal_weight_thesh1,X_train_mains,Y_train_mains)
predicted_values_test_t1,rmse_test_t1=regression_prediction(optimal_weight_thesh1,X_test_mains,Y_test_mains)
predicted_values_test_t2,rmse_test_t2=regression_prediction(optimal_weight_thesh2,X_test_mains,Y_test_mains)
predicted_values_test_t3,rmse_test_t3=regression_prediction(optimal_weight_thesh3,X_test_mains,Y_test_mains)
predicted_values_test_t4,rmse_test_t4=regression_prediction(optimal_weight_thesh4,X_test_mains,Y_test_mains)
predicted_values_test_t5,rmse_test_t5=regression_prediction(optimal_weight_thesh5,X_test_mains,Y_test_mains)

print('regression analysis completed')
#print(rmse_train)
#print(rmse_test)

#len(model.feature_importances_)

# Importing libraries for classification
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import parfit.parfit as pf
from sklearn import linear_model


# getting out workset ready
X = dataset_mains
Y = energy_source_data_features[YMaster]
logistic_workset = pd.concat([X,Y],axis=1)

# Class variable definition for spike alert, threshold of 150 set
logistic_workset['spike_alert'] = pd.cut(logistic_workset['Appliances'], bins=[0,150,float('Inf')], labels=['low', 'high'])    

# dropping the continuous variable
logistic_workset.drop(['Appliances'],axis=1,inplace = True)


# Splitting training and test set into desired ratio, here 77.5-22.5
# we make sure that we do stratifying sampling so that the class distribution is maintained in both train and test set
x_train_logistic, x_test_logistic, y_train_logistic, y_test_logistic = train_test_split(logistic_workset.iloc[:,:-1], logistic_workset['spike_alert'],
                                                stratify=logistic_workset['spike_alert'], 
                                                test_size=0.225)



# Intialising grid for learn rate and specifying attribute log for logistic regression
grid = {
    'alpha': [0.001,0.005,0.0075,0.0085, 0.009,0.0095, 0.01, 0.015, 0.020,0.05,0.1,0.15], # learning rate
        'loss': ['log'] # logistic regression,
}
paramGrid = ParameterGrid(grid)



# Getting best fit on test set using all features
bestModel, bestScore, allModels, allScores = pf.bestFit(SGDClassifier, paramGrid,
           x_train_logistic, y_train_logistic, x_test_logistic, y_test_logistic, 
           metric = roc_auc_score, scoreLabel = "AUC")

print(bestModel, bestScore)

# Getting best fit on train set using all features
bestModel_train, bestScore_train, allModels, allScores = pf.bestFit(SGDClassifier, paramGrid,
           x_train_logistic, y_train_logistic, x_train_logistic, y_train_logistic, 
           metric = roc_auc_score, scoreLabel = "AUC")

print(bestModel_train, bestScore_train)

# Experiment 4 and 5 for logistic regression

# Getting best fit on train set using all features
mySGDlr = linear_model.SGDClassifier(loss = 'log',alpha=0.01)
mySGDlr.fit(x_train_logistic,y_train_logistic)
#mySGDlr.score(x_train_logistic,y_train_logistic)

print(mySGDlr.score(x_test_logistic,y_test_logistic))
print(mySGDlr.score(x_train_logistic,y_train_logistic))

#Exp4
# Training and Predicting the model using 10 randomly selected features , train and test set accuracy
mySGDlr.fit(x_train_logistic[energy_source_data_features_sampled_random.columns],y_train_logistic)
print(mySGDlr.score(x_train_logistic[energy_source_data_features_sampled_random.columns],y_train_logistic))
print(mySGDlr.score(x_test_logistic[energy_source_data_features_sampled_random.columns],y_test_logistic))

# Exp5
# Training and Predicting the model using 10 chosen features,, train and test set accuracy
mySGDlr.fit(x_train_logistic[important_features],y_train_logistic)
print(mySGDlr.score(x_train_logistic[important_features],y_train_logistic))
print(mySGDlr.score(x_test_logistic[important_features],y_test_logistic))

#mySGDlr.coef_

print('End of assignment')
