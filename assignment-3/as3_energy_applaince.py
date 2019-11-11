# Importing required libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime as dt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from keras.utils import np_utils
import keras.metrics
from keras import optimizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier  #GBM algorithm
import matplotlib.pyplot as plt
from keras.layers import Dropout
import itertools
import scipy.spatial.distance


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



# Intial Feature Selection after removing nine variables after correlation analysis
XMaster_Updated=['lights',  'RH_1', 'T2', 'RH_2', 'T3',
        'T4',  'RH_6', 'T7', 'RH_7', 'T8',
       'RH_8', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
        'Tdewpoint', 'rv2', 'month',
       'is_weekend',  'time_of_day_encoded']

YMaster = ['spike_alert']

# # Class variable definition for spike alert, threshold of 150 set
energy_source_data_features['spike_alert'] = pd.cut(energy_source_data_features['Appliances'], bins=[0,120,float('Inf')], labels=[0, 1])    
print(energy_source_data_features['spike_alert'].value_counts())
# Feature Scaling
energy_indep_var = energy_source_data_features[XMaster_Updated]
energy_dep_var   = energy_source_data_features[YMaster]

scaler  = StandardScaler()
X_train_out, X_test_out, Y_train_out, Y_test_out = train_test_split(energy_indep_var,energy_dep_var,test_size=0.3,random_state=121)

X_train_out = scaler.fit_transform(X_train_out)
X_test_out  = scaler.fit_transform(X_test_out)


x_train=X_train_out
x_test=X_test_out
y_train=Y_train_out
y_test=Y_test_out
encoding_test_y = np_utils.to_categorical(y_test)
encoding_train_y = np_utils.to_categorical(y_train)


input_dim=x_train.shape[1]
# Neural Network with default features
# Creating a model
model = Sequential()
#model.add(Dropout(0.01, input_shape=(21,)))
# randomly chosen number of neurons
model.add(Dense(60,input_dim=input_dim))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))

model.add(Dense(2,activation="sigmoid"))

sgd = optimizers.SGD()

# Compiling model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Fitting the model
model.fit(x_train, encoding_train_y,epochs=20, batch_size=30)
predictions = model.predict_classes(x_test)

# Plotting ROC curve for baseline model
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
roc_auc = auc(fpr, tpr)
print(accuracy_score(y_train,model.predict_classes(x_train)))
print(confusion_matrix(y_train, model.predict_classes(x_train)))

print(accuracy_score(y_test,model.predict_classes(x_test)))
print(confusion_matrix(y_test, model.predict_classes(x_test)))
print(roc_auc)

# plotting the roc and generating confusion_matrix for basline model above
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw,label="area under curve = %1.2f" % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()



neurons = np.linspace(40, 100, 6, endpoint=True,dtype='int16')
 

# Vary neuron and get out of sample error and other important metrics

train_results = []
test_results = []
train_error = []
test_error = []

for neuron in neurons:
    
    model = Sequential()
    model.add(Dense(neuron,input_dim=input_dim))
    model.add(Dense(neuron))
    model.add(Dense(neuron))
    model.add(Dense(neuron))
    model.add(Dense(neuron,activation='sigmoid'))

    model.add(Dense(2))

    sgd = optimizers.SGD()

    # Compiling model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # Fitting the model
    model.fit(x_train, encoding_train_y,epochs=20, batch_size=20)
    train_pred = model.predict_classes(x_train)
    y_pred = model.predict_classes(x_test)

    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)
    train_error.append(1-accuracy_score(y_train, train_pred))
    
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    test_error.append(1-accuracy_score(y_test, y_pred))
    
# plotting AUC error for each neuron count
    
plt.subplot(2, 1, 1)
    
line1, = plt.plot(neurons, train_results, 'b', label='Train AUC')
line2, = plt.plot(neurons, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.ylabel('AUC score')
plt.xlabel('Neuron')
plt.show()

# plotting error rate for each neuron count
plt.subplot(2, 1, 2)

line1, = plt.plot(neurons, train_error, 'b', label='Train Error')
line2, = plt.plot(neurons, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('Neuron')
plt.show()


# tuning number of layers keeping neurons fixed
nueron=80
def create_model(hidden_layers=1,optimizer='adam', activation = 'sigmoid', nueron=80):
    # Initialize the constructor
    model = Sequential()
    # Add an input layer
    model.add(Dense(nueron, input_dim=input_dim))

    for i in range(hidden_layers):
      # Add one hidden layer
        model.add(Dense(nueron))

    # Add an output layer 
    model.add(Dense(1, activation=activation))
    #compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,epochs=15, batch_size=20)
#grid_result = grid.fit(X, Y)
hidden_layers=[1,2,3,4,5,6,7,8]
param_grid = dict(hidden_layers=hidden_layers)
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)
grid_result = grid.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# plotting cross validation error with repspect to number of layers    
plt.plot([1,2,3,4,5,6,7,8],[0.8544, 0.8539, 0.8513,0.8534,0.8523,0.8547,0.8530,0.8524])
#plt.plot(hidden_layers,cv_error)
plt.xlabel('Number of Hidden Layers')
plt.ylabel('CV - Accuracy Score')
plt.show()


# tuning activation functions
activation_functions=['tanh','sigmoid','softmax','exponential','softsign'] 

# Vary activation_function and get out of sample error and other important metrics

train_results = []
test_results = []
train_error = []
test_error = []


# calculating the roc and out of sample error metric for each paramerter value
for activation_function in activation_functions:
    
    model = Sequential()
    model.add(Dense(nueron,input_dim=input_dim))
    model.add(Dense(nueron,activation=activation_function))
    model.add(Dense(nueron,activation=activation_function))
    
    model.add(Dense(2,activation=activation_function))

    sgd = optimizers.SGD()

    # Compiling model
    model.compile(loss='squared_hinge', optimizer='sgd', metrics=['accuracy'])

    # Fitting the model
    model.fit(x_train, encoding_train_y,epochs=20, batch_size=20)
    train_pred = model.predict_classes(x_train)
    y_pred = model.predict_classes(x_test)

    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)
    train_error.append(1-accuracy_score(y_train, train_pred))
    
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    test_error.append(1-accuracy_score(y_test, y_pred))
    
# plotting AUC error for each activation_function
    
plt.subplot(2, 1, 1)
    
line1, = plt.plot(activation_functions, train_results, 'b', label='Train AUC')
line2, = plt.plot(activation_functions, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.ylabel('AUC score')
plt.xlabel('Activation Function')
plt.show()

# plotting error rate for each activation function
plt.subplot(2, 1, 2)

line1, = plt.plot(activation_functions, train_error, 'b', label='Train Error')
line2, = plt.plot(activation_functions, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('Activation Function')
plt.show()

loss_funcs=['cosine_proximity','hinge','logcosh','mean_squared_logarithmic_error','categorical_crossentropy','squared_hinge'] 
activation='softsign'
# Vary loss func  and get out of sample error and other important metrics

train_results = []
test_results = []
train_error = []
test_error = []

for loss_func in loss_funcs:
    
    model = Sequential()
    model.add(Dense(80,input_dim=input_dim))
    model.add(Dense(80,activation=activation))
    model.add(Dense(80,activation=activation))
    model.add(Dense(80,activation=activation))
 

    model.add(Dense(2))

    sgd = optimizers.SGD()
    #print(learn_rate)


    # Compiling model
    model.compile(loss=loss_func, optimizer='sgd', metrics=['accuracy'])

    # Fitting the model
    model.fit(x_train, encoding_train_y,epochs=15, batch_size=15)
    train_pred = model.predict_classes(x_train)
    y_pred = model.predict_classes(x_test)

    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)
    train_error.append(1-accuracy_score(y_train, train_pred))
    
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    test_error.append(1-accuracy_score(y_test, y_pred))
    

# ploting roc and out of sample error for each loss func iteration
loss_funcs=['cosine','hinge','logcosh','msqlgerr','crossent','sqhinge']
plt.subplot(2, 1, 1)
    
line1, = plt.plot(loss_funcs, train_results, 'b', label='Train AUC')
line2, = plt.plot(loss_funcs, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.ylabel('AUC score')
plt.xlabel('loss')
plt.show()

# plotting error rate for each loss func
plt.subplot(2, 1, 2)

line1, = plt.plot(loss_funcs, train_error, 'b', label='Train Error')
line2, = plt.plot(loss_funcs, test_error, 'r', label='Test Error')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('loss')
plt.show()



# Vary batch size  and get out of sample error and other important metrics

batch_sizes = [3,4,5,6,10,20]
 

# Vary batch size and get out of sample error and other important metrics

train_results = []
test_results = []
train_error = []
test_error = []

#Calucting roc and out of sample error

for batch_size in batch_sizes:
    
    model = Sequential()
    model.add(Dense(80,input_dim=input_dim))
    model.add(Dense(80,activation=activation))
    model.add(Dense(80,activation=activation))
    model.add(Dense(80,activation=activation))

    model.add(Dense(2))

    sgd = optimizers.SGD()

    # Compiling model
    model.compile(loss='squared_hinge', optimizer='sgd', metrics=['accuracy'])

    # Fitting the model
    model.fit(x_train, encoding_train_y,epochs=20, batch_size=batch_size)
    train_pred = model.predict_classes(x_train)
    y_pred = model.predict_classes(x_test)

    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)
    train_error.append(1-accuracy_score(y_train, train_pred))
    
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    test_error.append(1-accuracy_score(y_test, y_pred))
    
# plotting AUC error for each batch size
    
plt.subplot(2, 1, 1)
    
line1, = plt.plot(batch_sizes, train_results, 'b', label='Train AUC')
line2, = plt.plot(batch_sizes, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.ylabel('AUC score')
plt.xlabel('Batch Size')
plt.show()

# plotting error rate for each batch size
plt.subplot(2, 1, 2)

line1, = plt.plot(batch_sizes, train_error, 'b', label='Train Error')
line2, = plt.plot(batch_sizes, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('Batch Size')
plt.show()


# vary learn rate and get computation metrics

learn_rates = [0.01,0.05,0.1,0.15]
 
batch_size=3
# Vary learn_rate and get out of sample error and other important metrics

train_results = []
test_results = []
train_error = []
test_error = []



for learn_rate in learn_rates:
    
    model = Sequential()
    model.add(Dense(80,input_dim=input_dim))
    model.add(Dense(80,activation=activation))
    model.add(Dense(80,activation=activation))
    model.add(Dense(80,activation=activation))

    model.add(Dense(2))

    sgd = optimizers.SGD(learning_rate=learn_rate)
    print(learn_rate)


    # Compiling model
    model.compile(loss='squared_hinge', optimizer='sgd', metrics=['accuracy'])

    # Fitting the model
    model.fit(x_train, encoding_train_y,epochs=20, batch_size=batch_size)
    train_pred = model.predict_classes(x_train)
    y_pred = model.predict_classes(x_test)

    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)
    train_error.append(1-accuracy_score(y_train, train_pred))
    
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    test_error.append(1-accuracy_score(y_test, y_pred))
    
# plotting AUC error for each learn rate
    
plt.subplot(2, 1, 1)
    
line1, = plt.plot(learn_rates, train_results, 'b', label='Train AUC')
line2, = plt.plot(learn_rates, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.ylabel('AUC score')
plt.xlabel('Learn_Rate')
plt.show()

# plotting error rate for each learn rate
plt.subplot(2, 1, 2)

line1, = plt.plot(learn_rates, train_error, 'b', label='Train Error')
line2, = plt.plot(learn_rates, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('Learn_Rate')
plt.show()


epochs = [5,10,15,20,30,35]
learn_rate=0.10
 

# Vary epochs and get out of sample error and other important metrics

train_results = []
test_results = []
train_error = []
test_error = []
activation='softsign'
for epoch in epochs:
    
    model = Sequential()
    model.add(Dense(80,input_dim=input_dim))
    model.add(Dense(80,activation=activation))
    model.add(Dense(80,activation=activation))
    model.add(Dense(80,activation=activation))

    model.add(Dense(2))

    sgd = optimizers.SGD(learning_rate=learn_rate)
    print(learn_rate)


    # Compiling model
    model.compile(loss='squared_hinge', optimizer=sgd, metrics=['accuracy'])

    # Fitting the model
    model.fit(x_train, encoding_train_y,epochs=epoch, batch_size=3)
    train_pred = model.predict_classes(x_train)
    y_pred = model.predict_classes(x_test)

    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)
    train_error.append(1-accuracy_score(y_train, train_pred))
    
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    test_error.append(1-accuracy_score(y_test, y_pred))
    
# plotting AUC error for each learn rate
    
plt.subplot(2, 1, 1)
    
line1, = plt.plot(epochs, train_results, 'b', label='Train AUC')
line2, = plt.plot(epochs, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.ylabel('AUC score')
plt.xlabel('epochs')
plt.show()

# plotting error rate for each learn rate
plt.subplot(2, 1, 2)

line1, = plt.plot(epochs, train_error, 'b', label='Train Error')
line2, = plt.plot(epochs, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('epochs')
plt.show()

def create_model(optimizer='sgd', activation = 'tanh', nueron=80,learn_rate=0.10,loss='squared_hinge'):
    # Initialize the constructor
    model = Sequential()
    # Add an input layer
    model.add(Dense(nueron, input_dim=input_dim))
    model.add(Dense(nueron,activation=activation))
    model.add(Dense(nueron,activation=activation))
    model.add(Dense(nueron,activation=activation))

    
    # Add an output layer 
    model.add(Dense(1, activation=activation))
    #compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,epochs=20, batch_size=3)
#grid_result = grid.fit(X, Y)
nueron=[60,80]
loss=['squared_hinge','cosine_proximity']
activation=['softsign','tanh']
param_grid = dict(nueron=nueron,loss=loss,activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=3)
grid_result = grid.fit(x_train, y_train)

# summarize results ujsing grid search with a grid of params converging
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#Best: 0.848559 using {'activation': 'softsign', 'loss': 'cosine_proximity', 'nueron': 60}

model = Sequential()
activation='softsign'
#model.add(Dropout(0.01, input_shape=(21,)))
model.add(Dense(80,input_dim=input_dim))
model.add(Dense(80,activation=activation))
model.add(Dense(80,activation=activation))
model.add(Dense(80,activation=activation))

model.add(Dense(2,activation=activation))

sgd = optimizers.SGD(learning_rate=0.10)

# Compiling model
model.compile(loss='squared_hinge', optimizer=sgd, metrics=['accuracy'])

# Fitting the model
model.fit(x_train, encoding_train_y,epochs=50, batch_size=3)
predictions = model.predict_classes(x_test)

# Plotting ROC curve for tuned model after grid search
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
roc_auc = auc(fpr, tpr)
print(accuracy_score(y_train,model.predict_classes(x_train)))
print(confusion_matrix(y_train, model.predict_classes(x_train)))

print(accuracy_score(y_test,model.predict_classes(x_test)))
print(confusion_matrix(y_test, model.predict_classes(x_test)))
print(roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw,label="area under curve = %1.2f" % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()

error_trace=[0.4758,0.4406,0.4293,0.4232,0.4152,0.4093,0.403,0.4002,0.3936,0.3873,0.381,0.3766,0.3704,0.3651,0.3609,0.3557,0.3537,0.3477,0.3393,0.339,0.3343,0.3287,0.3268,0.3213,0.3213,0.3129,0.3127,0.3108,0.3093,0.3045,0.2988,0.2998,0.2984,0.2956,0.2936,0.2896,0.287,0.2861,0.2835,0.2833,0.278,0.2763,0.271,0.2724,0.2695,0.2686,0.2657,0.2696,0.2641,0.2621]
epochs = np.linspace(1, 50, 50, endpoint=True,dtype='int16')
plt.plot(epochs,error_trace)
plt.xlabel('Epochs')
plt.ylabel('Loss_Calculation')
plt.title('Loss with repect to Epochs')
plt.legend(loc="lower right")
plt.show()



######################################
#####################################
### kNN Model

clf = KNeighborsClassifier()


#training the model on default features
clf = clf.fit(x_train,y_train)


#Predict the response for train dataset
pred_out_knn = clf.predict(x_train)
print(accuracy_score(y_train, pred_out_knn))
print(confusion_matrix(y_train, pred_out_knn))


# Predict the response for test set
pred_out_knn_test = clf.predict(x_test)
print(accuracy_score(y_test, pred_out_knn_test))
print(confusion_matrix(y_test, pred_out_knn_test))


# roc and conf matrix for basline  model

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_out_knn_test, pos_label=1)
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw,label="area under curve = %1.2f" % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()


# In[11]:



# Vary number of k neighbors and generate learning curve

n_estimators = np.linspace(1, 15, 15, endpoint=True,dtype='int16')
train_results = []
test_results = []
train_error = []
test_error = []

# 

for n in n_estimators:
    dt = KNeighborsClassifier(n_neighbors =n)
    dt.fit(x_train, y_train)
    train_pred = dt.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)
    train_error.append(1-accuracy_score(y_train, train_pred))
    
    y_pred = dt.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    test_error.append(1-accuracy_score(y_test, y_pred))
    
# ROC plot and out of sample error after varing k
    
plt.subplot(2, 1, 1)
    
line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')
line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('AUC score')
plt.xlabel('k-neighbors')
plt.show()

plt.subplot(2, 1, 2)

line1, = plt.plot(n_estimators, train_error, 'b', label='Train Error')
line2, = plt.plot(n_estimators, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('k-neighbors')
plt.show()



# In[12]:

# Vary number of estimators and generate learning curve
weights =['uniform','distance']
train_results = []
test_results = []
train_error = []
test_error = []

for n in weights:
    dt = KNeighborsClassifier(n_neighbors =3,weights=n)
    dt.fit(x_train, y_train)
    train_pred = dt.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)
    train_error.append(1-accuracy_score(y_train, train_pred))
    
    y_pred = dt.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    test_error.append(1-accuracy_score(y_test, y_pred))
    
# ROC plot
    
plt.subplot(2, 1, 1)
    
line1, = plt.plot(weights, train_results, 'b', label='Train AUC')
line2, = plt.plot(weights, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('AUC score')
plt.xlabel('weights')
plt.show()

plt.subplot(2, 1, 2)

line1, = plt.plot(weights, train_error, 'b', label='Train Error')
line2, = plt.plot(weights, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('weights')
plt.show()



# In[12]:


# Vary number of estimators and generate learning curve
algorithm=['auto', 'ball_tree','kd_tree','brute']
train_results = []
test_results = []
train_error = []
test_error = []


for n in algorithm:
    dt = KNeighborsClassifier(n_neighbors =3,weights='distance',algorithm=n)
    dt.fit(x_train, y_train)
    train_pred = dt.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)
    train_error.append(1-accuracy_score(y_train, train_pred))
    
    y_pred = dt.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    test_error.append(1-accuracy_score(y_test, y_pred))
    
# ROC plot
    
plt.subplot(2, 1, 1)
    
line1, = plt.plot(algorithm, train_results, 'b', label='Train AUC')
line2, = plt.plot(algorithm, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('AUC score')
plt.xlabel('algorithm')
plt.show()

plt.subplot(2, 1, 2)

line1, = plt.plot(algorithm, train_error, 'b', label='Train Error')
line2, = plt.plot(algorithm, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('algorithm')
plt.show()



# In[12]:


# Vary number of estimators and generate learning curve
metric=['minkowski', 'cosine', 'euclidean','manhattan']
train_results = []
test_results = []
train_error = []
test_error = []


for n in metric:
    dt = KNeighborsClassifier(n_neighbors =3,weights='distance',algorithm='auto',metric=n)
    dt.fit(x_train, y_train)
    train_pred = dt.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)
    train_error.append(1-accuracy_score(y_train, train_pred))
    
    y_pred = dt.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    test_error.append(1-accuracy_score(y_test, y_pred))
    
# ROC plot
    
plt.subplot(2, 1, 1)
    
line1, = plt.plot(metric, train_results, 'b', label='Train AUC')
line2, = plt.plot(metric, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('AUC score')
plt.xlabel('metric-distance')
plt.show()

plt.subplot(2, 1, 2)

line1, = plt.plot(metric, train_error, 'b', label='Train Error')
line2, = plt.plot(metric, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('metric-distance')
plt.show()



# Optimal model after hyperparam tuning


clf = KNeighborsClassifier(n_neighbors=1,algorithm='auto',metric='manhattan')


#training the model on default features
clf = clf.fit(x_train,y_train)


#Predict the response for train dataset
pred_out_knn = clf.predict(x_train)
print(accuracy_score(y_train, pred_out_knn))
print(confusion_matrix(y_train, pred_out_knn))


# Predict the response for test set
pred_out_knn_test = clf.predict(x_test)
print(accuracy_score(y_test, pred_out_knn_test))
print(confusion_matrix(y_test, pred_out_knn_test))




# Since the baseline model is without pruning we get the max depth as 33
#print(clf.tree_.max_depth)

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_out_knn_test, pos_label=1)
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw,label="area under curve = %1.2f" % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()

