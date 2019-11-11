#Importing desired packages
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
from sklearn.feature_selection import SelectKBest, f_classif



# Data Preprocessing
def data_preprocess(kickstarter_source_dataset):

    kickstarter_source_dataset['state'].value_counts()

    # Getting states suspended, cancelled, successful, failed
    kickstarter_projects = kickstarter_source_dataset[(kickstarter_source_dataset['state'] == 'failed')|(kickstarter_source_dataset['state']
    == 'successful')|(kickstarter_source_dataset['state'] == 'canceled')|(kickstarter_source_dataset['state'] == 'suspended')]

    # Populating the states canceled, suspended, failed as successful
    kickstarter_projects.loc[kickstarter_projects['state'] == 'canceled', 'state'] = 'unsuccessful'
    kickstarter_projects.loc[kickstarter_projects['state'] == 'suspended', 'state'] = 'unsuccessful'
    kickstarter_projects.loc[kickstarter_projects['state'] == 'failed', 'state'] = 'unsuccessful'

    # We now have two states for the target variable, successful or unsuccessful
    kickstarter_projects['state'].value_counts()

    ((kickstarter_projects.isnull() | kickstarter_projects.isna()).sum() * 100 / kickstarter_projects.index.size).round(2)
    # We see that 234 missing values in usd pledged, we populate these missing values using values from usd_pledged_real
    kickstarter_projects['usd pledged'].fillna(kickstarter_projects.usd_pledged_real, inplace=True)
    # Removing projects with goal less than 100 dollors
    kickstarter_projects = kickstarter_projects[kickstarter_projects['goal']>100]

    return kickstarter_projects


#Function for counting syllables in project name
# Semantic
# If first word vowel add one syllable
# Increase counter if vowel is followed by a consonant
# Set minimum syllable value as one.
def syllable_count(project_name):

    word=str(project_name).lower()
    count=0
    vowels='aeiou'
    word=str(word)
    first=word[:1]
    if first in vowels:
        count+=1

    for index in range(1,len(word)):

        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
    return count

                                                                                                                                                 # Feature engineering
def feature_engineering(kickstarter_workset):

    # Invoking function to calculate syllables in the project name
    kickstarter_workset["syllable_count"]   = kickstarter_workset["name"].apply(lambda x: syllable_count(x))

    #Converting launched and deadline values to datetime pandas objects
    kickstarter_workset['launched']         = pd.to_datetime(kickstarter_workset['launched'])
    kickstarter_workset['deadline']         = pd.to_datetime(kickstarter_workset['deadline'])

    # Getting values for launched month, launched week, launched day
    kickstarter_workset["launched_month"]   = kickstarter_workset["launched"].dt.month
    kickstarter_workset["launched_week"]    = kickstarter_workset["launched"].dt.week
    kickstarter_workset["launched_day"]     = kickstarter_workset["launched"].dt.weekday
    kickstarter_workset['launched_year']    = kickstarter_workset['launched'].dt.year
    kickstarter_workset['launched_quarter'] = kickstarter_workset['launched'].dt.quarter

    # Marking the flag if the launch day falls on the weekend
    kickstarter_workset["is_weekend"]       = kickstarter_workset["launched_day"].apply(lambda x: 1 if x > 4 else 0)
    # Number of words in the name of the projects
    kickstarter_workset["num_words"]        = kickstarter_workset["name"].apply(lambda x: len(str(x).split()))
    # Duration calculation using the differnece between launching date and deadline
    kickstarter_workset["duration"]         = kickstarter_workset["deadline"] - kickstarter_workset["launched"]
    kickstarter_workset["duration"]         = kickstarter_workset["duration"].apply(lambda x: int(str(x).split()[0]))
    
      # Competition evaluation
    # This variable calculates the number of projects launched in the same week belonging to the same category
    kickstarter_workset['launched_year_week_category']        = kickstarter_workset['launched_year'].astype(str)+"_"+kickstarter_workset['launched_week'].astype(str)+"_    "+kickstarter_workset['main_category'].astype(str)
    kickstarter_workset['launched_year_week']                 = kickstarter_workset['launched_year'].astype(str)+"_"+kickstarter_workset['launched_week'].astype(str)

    # Getting average number of projects launched per week for each of the main categories category
    kickstarter_workset['week_count']                         = kickstarter_workset.groupby('main_category')['launched_year_week'].transform('nunique')
    kickstarter_workset['project_count_category']             = kickstarter_workset.groupby('main_category')['ID'].transform('count')
    kickstarter_workset['weekly_average_category']            = kickstarter_workset['project_count_category']/kickstarter_workset['week_count']
    kickstarter_workset_category_week=kickstarter_workset[['main_category','weekly_average_category']].drop_duplicates()

    #Calculating number of projects launched for a combination of (year,week) for each main category
    kickstarter_workset['weekly_category_launch_count']       = kickstarter_workset.groupby('launched_year_week_category')['ID'].transform('count')

    #Competiton quotient
    kickstarter_workset['competition_quotient']               = kickstarter_workset['weekly_category_launch_count']/kickstarter_workset['weekly_average_category']

    # Goal Level
    # In this feature we compare the project goal with the mean goal for the category it belongs, the mean goal for the category is used as the normaliation coefficient
    kickstarter_workset['mean_category_goal']                 = kickstarter_workset.groupby('main_category')['goal'].transform('mean')
    kickstarter_workset['goal_level']                         = kickstarter_workset['goal']   / kickstarter_workset['mean_category_goal']

    #Duration Level
    # In this feature we compare the project duration with the mean duration for the category with the duration of the given project
    kickstarter_workset['mean_category_duration']             = kickstarter_workset.groupby('main_category')['duration'].transform('mean')
    kickstarter_workset['duration_level']                     = kickstarter_workset['duration']   / kickstarter_workset['mean_category_duration']

    
    #Binning the Competition Quotient
    bins_comp_quot                                     = np.array([0,0.25,1,1.5,2.5,10])
    kickstarter_workset["competition_quotient_bucket"] = pd.cut(kickstarter_workset.competition_quotient, bins_comp_quot)

    #Binning the Duration Level
    bins_duration_level                                = np.array([0,0.25,1,2,4])
    kickstarter_workset["duration_level_bucket"]       = pd.cut(kickstarter_workset.duration_level, bins_duration_level)                           
    # Binning the USD Goal level
    bins_goal_level                                    = np.array([0,0.5,1.5,5,200])
    kickstarter_workset['goal_level_bucket']           = pd.cut(kickstarter_workset.goal_level, bins_goal_level)                                   
    # Calculating the average amount spent per backer

    kickstarter_workset['average_amount_per_backer']   = kickstarter_workset['usd pledged']/kickstarter_workset['backers']

    # Marking currency as dollor, euro, gbp and others , this variable is strongly correlated with the country of launch
    kickstarter_workset['currency_usd_flag']   = np.where(kickstarter_workset['currency'] == 'USD',1,0)

    # Discarding some features that were created in the intermediate steps and retaining the remaining features
    kickstarter_workset=kickstarter_workset[['ID', 'name', 'category', 'main_category', 'currency', 'deadline','goal',                                     'launched', 'pledged', 'state','backers', 'country',  'usd pledged', 'syllable_count', 'launched_month',
        'launched_day', 'launched_year','launched_quarter', 'is_weekend','num_words',
        'duration','competition_quotient','goal_level', 'duration_level', 'competition_quotient_bucket',
        'duration_level_bucket', 'goal_level_bucket',                                                                                                      'average_amount_per_backer','currency_usd_flag']]

   # kickstarter_workset=kickstarter_workset.columns.str.replace(' ','_')

    return kickstarter_workset


#import kickstarter_success_mains as ks
#console for global variables and functions call
#config_file_name = 'loc_config.ini'
#print(config_file_name)
# Getting the two source datasets
# Encoding ISO-8859-1 used since some of the project names have non ascii characters
energy_source_data=pd.read_csv('ks-projects-201801.csv')
kickstarter_workset=data_preprocess(energy_source_data)
kickstarter_workset=feature_engineering(kickstarter_workset)
#kickstarter_workset=ks.prepare_model_data(kickstarter_workset)


#def prepare_model_data(kickstarter_workset):
feature_categorical = ['main_category', 'launched_year', 'launched_quarter','is_weekend',
        'duration_level_bucket']

feature_numeric = ['average_amount_per_backer', 'goal_level', 'competition_quotient', 'syllable_count','backers','duration']
#kickstarter_workset['goal_level_bucket']

features_main=feature_categorical+feature_numeric

#kickstarter_workset=kickstarter_workset.sample(n=4000)

for col in feature_categorical:
        kickstarter_workset[col] = kickstarter_workset[col].astype('category')


kick_projects_ip = pd.get_dummies(kickstarter_workset[features_main],columns = feature_categorical)

kick_projects_ip = kick_projects_ip.loc[:,~kick_projects_ip.columns.duplicated()]

# for column in kick_projects_ip.columns:
#     if kick_projects_ip[column].dtype == type(object):
#         le = LabelEncoder()
#         dataset[column] = le.fit_transform(dataset[column])

kick_projects_ip['state']=kickstarter_workset['state']

kick_projects_ip=kick_projects_ip.dropna()
kick_projects_ip=kick_projects_ip[~kick_projects_ip.isin([np.nan, np.inf, -np.inf]).any(1)]

#kickstarter_workset= kick_projects_ip.dropna()
print(kick_projects_ip.isnull().values.any())
kick_projects_ip=kick_projects_ip.sample(30000)
print('dataset sampled')
codes = {'successful':0, 'unsuccessful':1}
kick_projects_ip['state'] = kick_projects_ip['state'].map(codes)
#kick_projects_ip['state'] = pd.to_numeric(kick_projects_ip['state'], errors='coerce')

y=kick_projects_ip['state']
y = pd.DataFrame(y,columns = ['state'])
X=kick_projects_ip[kick_projects_ip.columns]
X=X.drop('state', 1)


# pick top 14 features
X = SelectKBest(f_classif, k=14).fit_transform(X, y)


# 
print(X.shape)
print(y.shape)


# Stratified split of train and test set
X_train_out, X_test_out, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)
#print(n)

x_train=X_train_out.round(2)
x_test=X_test_out.round(2)



# Scaling using mean and standard deviation
scaler  = StandardScaler()
X_train_out = scaler.fit_transform(X_train_out)
X_test_out  = scaler.fit_transform(X_test_out)

print('scaling done')


encoding_test_y = np_utils.to_categorical(y_test)
encoding_train_y = np_utils.to_categorical(y_train)


input_dim=x_train.shape[1]
# Neural Network with default features
# Creating a model
# Baseline model
model = Sequential()
#model.add(Dropout(0.01, input_shape=(21,)))
# Default randomly chosen parameters with no consideration
model.add(Dense(9, input_dim=input_dim))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
#model.add(Dense(6, activation = 'softmax')) 

model.add(Dense(2))

sgd = optimizers.SGD()

# Compiling model using cosine proximity
model.compile(loss='cosine_proximity', optimizer='sgd', metrics=['accuracy'])

# Fitting the model
model.fit(x_train, encoding_train_y,epochs=15, batch_size=20)
predictions = model.predict_classes(x_test)

# Plotting ROC curve for baseline model
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
roc_auc = auc(fpr, tpr)
print(accuracy_score(y_train,model.predict_classes(x_train)))
print(confusion_matrix(y_train, model.predict_classes(x_train)))

print(accuracy_score(y_test,model.predict_classes(x_test)))
print(confusion_matrix(y_test, model.predict_classes(x_test)))
print(roc_auc)
# roC curve plot
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

# implementing cross validation for number of hidden laters; 3 folds layers 1 to 8
nueron=20
def create_model(hidden_layers=1,optimizer='adam', activation = 'sigmoid', nueron=nueron):
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
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=3)
grid_result = grid.fit(x_train, y_train)

# summarize results with grid search
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Plotting accuracy scores for each layer    
plt.plot([1,2,3,4,5,6,7,8],[0.805875,0.789167,0.770375,0.787583,0.779667,0.808917,0.786292,0.76725])
plt.xlabel('Number of Hidden Layers')
plt.ylabel('CV - Accuracy Score')
plt.show()




# In this step we vary neurons, 20,40,60,80,100
neurons = np.linspace(20, 100, 5, endpoint=True,dtype='int16')
 

# Vary neuron and get out of sample error and other important metrics

train_results = []
test_results = []
train_error = []
test_error = []

# 
for neuron in neurons:
    
    model = Sequential()
    #model.add(Dense(neuron,input_dim=input_dim))

    model.add(Dense(neuron, input_dim=input_dim))
    model.add(Dense(neuron))
    model.add(Dense(neuron))
    model.add(Dense(neuron))
    model.add(Dense(neuron))
    model.add(Dense(neuron))


    model.add(Dense(neuron))


    model.add(Dense(2))

    sgd = optimizers.SGD()

    # Compiling model
    model.compile(loss='cosine_proximity', optimizer='sgd', metrics=['accuracy'])

    # Fitting the model
    model.fit(x_train, encoding_train_y,epochs=15, batch_size=20)
    train_pred = model.predict_classes(x_train)
    y_pred = model.predict_classes(x_test)

    # for each run out of sample AUC and test error calculated
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)
    train_error.append(1-accuracy_score(y_train, train_pred))
    
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    test_error.append(1-accuracy_score(y_test, y_pred))
    
# plotting auc for each neuron count
    
plt.subplot(2, 1, 1)
    
line1, = plt.plot(neurons, train_results, 'b', label='Train AUC')
line2, = plt.plot(neurons, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.ylabel('AUC score')
plt.xlabel('Neuron')
plt.show()

# plotting error rate for each neuron count size
plt.subplot(2, 1, 2)

line1, = plt.plot(neurons, train_error, 'b', label='Train Error')
line2, = plt.plot(neurons, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('Neuron')
plt.show()



## Implementing the same analysis for 5 differnent activation functions
activation_functions=['tanh','sigmoid','softmax','exponential','softsign'] 

# Vary activation_function and get out of sample error and other important metrics

train_results = []
test_results = []
train_error = []
test_error = []
neuron=40

for activation_function in activation_functions:
    model = Sequential()
    #model.add(Dense(neuron,input_dim=input_dim))

    model.add(Dense(neuron, input_dim=input_dim))
    model.add(Dense(neuron,activation=activation_function))
    model.add(Dense(neuron,activation=activation_function))
    model.add(Dense(neuron,activation=activation_function))
    model.add(Dense(neuron,activation=activation_function))
    model.add(Dense(neuron,activation=activation_function))
    model.add(Dense(neuron,activation=activation_function))

    
    model.add(Dense(2))

    sgd = optimizers.SGD()

    # Compiling model
    model.compile(loss='cosine_proximity', optimizer='sgd', metrics=['accuracy'])

    # Fitting the model
    model.fit(x_train, encoding_train_y,epochs=20, batch_size=20)
    train_pred = model.predict_classes(x_train)
    y_pred = model.predict_classes(x_test)

    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)
    train_error.append(1-accuracy_score(y_train, train_pred))
    
    # out of sample error evaluations
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # Add auc score to previous test results
    test_results.append(roc_auc)
    test_error.append(1-accuracy_score(y_test, y_pred))
    
# plotting AUC error for each activation fucntion
    
plt.subplot(2, 1, 1)
    
line1, = plt.plot(activation_functions, train_results, 'b', label='Train AUC')
line2, = plt.plot(activation_functions, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.ylabel('AUC score')
plt.xlabel('activation_functions')
plt.show()

# plotting error rate for each activation function
plt.subplot(2, 1, 2)

line1, = plt.plot(activation_functions, train_error, 'b', label='Train Error')
line2, = plt.plot(activation_functions, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('activation_functions')
plt.show()

# implementing the same analysis for loss functions, we take 6 different values and calculate out of sample errors and out of sample auc
loss_funcs=['cosine_proximity','hinge','logcosh','mean_squared_logarithmic_error','categorical_crossentropy','squared_hinge'] 
activation='softsign'
# Vary learn_rate and get out of sample error and other important metrics

train_results = []
test_results = []
train_error = []
test_error = []

# for each loss fucntion
for loss_func in loss_funcs:
    
    model = Sequential()
    #model.add(Dense(neuron,input_dim=input_dim))

    model.add(Dense(neuron, input_dim=input_dim))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))

    
    model.add(Dense(2))

    sgd = optimizers.SGD()

    # Compiling model
    model.compile(loss=loss_func, optimizer='sgd', metrics=['accuracy'])

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
    
# plotting AUC error for each loss function
    

    
loss_funcs=['cosine','hinge','logcosh','msqlgerr','crossent','sqhinge']
plt.subplot(2, 1, 1)
    
line1, = plt.plot(loss_funcs, train_results, 'b', label='Train AUC')
line2, = plt.plot(loss_funcs, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.ylabel('AUC score')
plt.xlabel('loss')
plt.show()

# plotting error rate for each loss function
plt.subplot(2, 1, 2)

line1, = plt.plot(loss_funcs, train_error, 'b', label='Train Error')
line2, = plt.plot(loss_funcs, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('loss')
plt.show()


batch_sizes = [3,4,5,6,10,20]
 

# Vary batch size and get out of sample error and other important metrics

train_results = []
test_results = []
train_error = []
test_error = []

for batch_size in batch_sizes:
    
    model = Sequential()
    model.add(Dense(neuron, input_dim=input_dim))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))

    model.add(Dense(2))

    sgd = optimizers.SGD()

    # Compiling model
    model.compile(loss='mean_squared_logarithmic_error', optimizer='sgd', metrics=['accuracy'])

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


learn_rates = [0.01,0.02,0.03]
 
batch_size=6
# Vary learn_rate and get out of sample error and other important metrics

train_results = []
test_results = []
train_error = []
test_error = []

for learn_rate in learn_rates:
    
    model = Sequential()
    model.add(Dense(80,input_dim=input_dim))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))
    model.add(Dense(neuron,activation=activation))

    model.add(Dense(2))

    sgd = optimizers.SGD(learning_rate=learn_rate)
    print(learn_rate)


    # Compiling model
    model.compile(loss='mean_squared_logarithmic_error', optimizer=sgd, metrics=['accuracy'])

    # Fitting the model
    model.fit(x_train, encoding_train_y,epochs=10, batch_size=batch_size)
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


epochs = [5,15,20,35]
learn_rate=0.01
 

# Vary epochs and get out of sample error and other important metrics

train_results = []
test_results = []
train_error = []
test_error = []

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
    model.compile(loss='mean_squared_logarithmic_error', optimizer=sgd, metrics=['accuracy'])

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
    
# plotting AUC error for each epoch
    
plt.subplot(2, 1, 1)
    
line1, = plt.plot(epochs, train_results, 'b', label='Train AUC')
line2, = plt.plot(epochs, test_results, 'r', label='Test AUC')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.ylabel('AUC score')
plt.xlabel('epochs')
plt.show()

# plotting error rate for each epoch
plt.subplot(2, 1, 2)

line1, = plt.plot(epochs, train_error, 'b', label='Train Error')
line2, = plt.plot(epochs, test_error, 'r', label='Test Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylabel('Error Rate')
plt.xlabel('epochs')
plt.show()

Hyper tuning comination of params using grid search cv

def create_model(optimizer='sgd', activation = 'tanh', nueron=80,learn_rate=0.01,loss='squared_hinge'):
    # Initialize the constructor
    model = Sequential()
    # Add an input layer
    model.add(Dense(nueron, input_dim=input_dim))
    model.add(Dense(nueron,activation=activation))
    model.add(Dense(nueron,activation=activation))
    model.add(Dense(nueron,activation=activation))
    model.add(Dense(nueron,activation=activation))
    model.add(Dense(nueron,activation=activation))
    model.add(Dense(nueron,activation=activation))

    
    # Add an output layer 
    model.add(Dense(1, activation=activation))
    #compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,epochs=10, batch_size=5)
#grid_result = grid.fit(X, Y)
nueron=[40,60,80]
loss=['squared_hinge','cosine_proximity']
activation=['softsign','tanh']
param_grid = dict(nueron=nueron,loss=loss,activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=3)
grid_result = grid.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    


# Best: 0.848559 using {'activation': 'softsign', 'loss': 'cosine_proximity', 'nueron': 60}

model = Sequential()
activation='softsign'
#model.add(Dropout(0.01, input_shape=(21,)))
model.add(Dense(80,input_dim=input_dim))
model.add(Dense(80,activation=activation))
model.add(Dense(80,activation=activation))
model.add(Dense(80,activation=activation))

model.add(Dense(2,activation=activation))

sgd = optimizers.SGD(learning_rate=0.01)

# Compiling model
model.compile(loss='mean_squared_logarithmic_error', optimizer='sgd', metrics=['accuracy'])

# Fitting the model
model.fit(x_train, encoding_train_y,epochs=10, batch_size=2)
predictions = model.predict_classes(x_test)

# Plotting ROC curve for baseline model
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
roc_auc = auc(fpr, tpr)
print(accuracy_score(y_train,model.predict_classes(x_train)))
print(confusion_matrix(y_train, model.predict_classes(x_train)))

print(accuracy_score(y_test,model.predict_classes(x_test)))
print(confusion_matrix(y_test, model.predict_classes(x_test)))
print(roc_auc)


# Plotting ROC curve for hyper tuned model
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

# loss function trace after each epoch
# plotting loss trace
error_trace=[0.4758,0.4406,0.4293,0.4232,0.4152,0.4093,0.403,0.4002,0.3936,0.3873]
epochs = np.linspace(1, 10, 10, endpoint=True,dtype='int16')
plt.plot(epochs,error_trace)
plt.xlabel('Epochs')
plt.ylabel('Loss_Calculation')
plt.title('Loss with repect to Epochs')
plt.legend(loc="lower right")
plt.show()


############################
############################
# KNN
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




# ROC curve for basline knn model
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



# Vary number of neighbors and generate learning curve

n_estimators = np.linspace(1, 20, 20, endpoint=True,dtype='int16')
train_results = []
test_results = []
train_error = []
test_error = []

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
    
# ROC plot for AUC and confusion matrix
    
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



# Vary the weight options and generate learning curve
weights =['uniform','distance']
train_results = []
test_results = []
train_error = []
test_error = []


# Plot out of sample error and AUC curve
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




# Vary algorithm opttions and generate learning curve
algorithm=['auto', 'ball_tree','kd_tree','brute']
train_results = []
test_results = []
train_error = []
test_error = []

# Plot rOC and out of sample error with varying parameter to tune it

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





# Vary metric options and generate learning curve
metric=['minkowski', 'cosine', 'euclidean','manhattan']
train_results = []
test_results = []
train_error = []
test_error = []


# Plot rOC and out of sample error with varying parameter to tune it
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


# Grid search CV
# Joint convergence of all params

params = {'n_neighbors':[3,6,9,12,15,18,20],
          'weights':['uniform', 'distance'],
        
          'metric':['minkowski', 'euclidean','manhattan']
         }

scoring = {'AUC': 'roc_auc'}
clf = GridSearchCV(KNeighborsClassifier(), params, cv=3)
clf.fit(x_test,y_test)

print(clf.best_params_)
print(len(clf.cv_results_['params']))
print(clf.best_estimator_)




clf = KNeighborsClassifier(n_neighbors=20,algorithm='auto',metric='manhattan')


#training the obtained model post gridsearch
clf = clf.fit(x_train,y_train)


#Predict the response for train dataset
pred_out_knn = clf.predict(x_train)
print(accuracy_score(y_train, pred_out_knn))
print(confusion_matrix(y_train, pred_out_knn))


# Predict the response for test set
pred_out_knn_test = clf.predict(x_test)
print(accuracy_score(y_test, pred_out_knn_test))
print(confusion_matrix(y_test, pred_out_knn_test))



# Plotting ROC

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




# In[ ]:



