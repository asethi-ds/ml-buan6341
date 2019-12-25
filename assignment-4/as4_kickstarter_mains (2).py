
# Section 0
# Importing Datasets
#Importing desired packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime as dt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt  
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.decomposition import FastICA
from sklearn import random_projection
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from keras.utils import np_utils
import keras.metrics
from keras import optimizers
from sklearn import neighbors
import matplotlib.pyplot as plt
from keras.layers import Dropout
from yellowbrick.cluster import InterclusterDistance
from mpl_toolkits import mplot3d
import plotly.express as px
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import davies_bouldin_score 
from statistics import stdev


# In[27]:



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
kick_projects_ip=kick_projects_ip.sample(10000)
print('dataset sampled')
codes = {'successful':0, 'unsuccessful':1}
kick_projects_ip['state'] = kick_projects_ip['state'].map(codes)
#kick_projects_ip['state'] = pd.to_numeric(kick_projects_ip['state'], errors='coerce')

y=kick_projects_ip['state']
y = pd.DataFrame(y,columns = ['state'])
X=kick_projects_ip[kick_projects_ip.columns]
#X=X.drop('state', 1)
from sklearn.preprocessing import LabelEncoder


# In[46]:





# In[29]:



#relevant_columns=['average_amount_per_backer','goal_level','competition_quotient','competition_quotient','backers','duration','is_weekend_1','state']

le = LabelEncoder()
X['state']= le.fit_transform(X['state']) 

#X=X[relevant_columns]
energy_source_data_features=X
# Stratified split of train and test set
X_train_out, X_test_out, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)
#print(n)

x_train=X_train_out.round(2)
x_test=X_test_out.round(2)
X.columns


# In[30]:


cost =[] 
for i in range(1, 15): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(X) 

    # calculates squared error 
    # for the clustered points 
    cost.append(KM.inertia_)

# plot the cost against K values 
plt.plot(range(1, 15), cost, color ='g', linewidth ='3') 
plt.xlabel("Value of K") 
plt.ylabel("Sqaured Error (Cost)") 

plt.show() # clear the plot 

# Getting cluster distance yellowbricks vizualisation
km = KMeans(n_clusters = 4, max_iter = 500)
km.fit(X)
kmeans_clusters=km.labels_
kmeans_centers=km.cluster_centers_
energy_source_data_features['kmeans_clusters']=kmeans_clusters

# Scaled plot of intercluster distances
visualizer = InterclusterDistance(km)
visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show() 


# principal components for cluster plot
pca = PCA(n_components=2)
principalComponents2 = pca.fit_transform(X)


# K means plot rudimentoary
plot_set=pd.DataFrame()
plot_set['x']=principalComponents2[:,0]
plot_set['y']=principalComponents2[:,1]
plot_set['label']=energy_source_data_features['kmeans_clusters']
x = plot_set['x']
y = plot_set['y']
cluster = plot_set['label']  # Labels of cluster 0 to 3
ax = sns.scatterplot(x="x", y="y", hue="label", style="label", data=plot_set)


# fig = plt.figure()
# ax = fig.add_subplot(111)
# scatter = ax.scatter(x,y,
#                      c=cluster,s=2)
# ax.set_title('K-Means Clustering scater plot')
# ax.set_xlabel('pc1')
# ax.set_ylabel('pc2')
# plt.colorbar(scatter)



# get inter cluster distance for rudimentary kmeans
dists = euclidean_distances(km.cluster_centers_)
tri_dists = dists[np.triu_indices(4, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
print('average inter cluster distance')
print(avg_dist)
print('dv_score kmeans')
# get DV score for rudimentary kmeans
print(davies_bouldin_score(X, kmeans_clusters)) 


# Frequency distribution of k means clusters
energy_source_data_features['kmeans_clusters'].value_counts()
# Cluster size plot for k means clusters
sizes=energy_source_data_features['kmeans_clusters'].value_counts()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
clust = ['Cluster0', 'Cluster1', 'Cluster2', 'Cluster3']

plt.xlabel('Clusters', fontsize=14)
plt.ylabel('Size of Clusters', fontsize=14)

ax.bar(clust,sizes)
plt.show()

kmeans_rudimentary_variation_score=stdev(sizes)
print('size variation score')
kmeans_rudimentary_variation_score


# In[31]:


#Section 3
### GMM Rudimentary version

gmm = GMM(n_components=4).fit(X)
gmm_clusters = gmm.predict(X)
energy_source_data_features['em_clusters']=gmm_clusters
gmm_centers=gmm.means_


# principal components for cluster plot gmm rudimentary
pca = PCA(n_components=2)
principalComponents2 = pca.fit_transform(X)


plot_set=pd.DataFrame()
plot_set['x']=principalComponents2[:,0]
plot_set['y']=principalComponents2[:,1]
plot_set['label']=energy_source_data_features['em_clusters']
x = plot_set['x']
y = plot_set['y']
cluster = plot_set['label']  # Labels of cluster 0 to 3

ax = sns.scatterplot(x="x", y="y", hue="label", style="label", data=plot_set)



# find inter cluster distance gmm_centers
dists = euclidean_distances(gmm_centers)
tri_dists = dists[np.triu_indices(4, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
print('average inter cluster distance')
print(avg_dist)
print('dv_score gmm')
print(davies_bouldin_score(X, gmm_clusters)) 

# clusterwise distribution of frequency gmm rudimentary
energy_source_data_features['em_clusters'].value_counts()

sizes=energy_source_data_features['em_clusters'].value_counts()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
clust = ['Cluster0', 'Cluster1', 'Cluster2', 'Cluster3']

plt.xlabel('Clusters', fontsize=14)
plt.ylabel('Size of Clusters', fontsize=14)

ax.bar(clust,sizes)
plt.show()



# # get mean of some relevant variables for GMM rudimentary for each cluster
# average_lights=gmm_centers[:,1]
# average_outside_temp=gmm_centers[:,20]
# average_hour=gmm_centers[:,32]
# average_outside_humidity=gmm_centers[:,22]

# # Plot means using a multiline graph
# x = np.arange(4)

# #plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])

# plt.plot(x, average_lights,marker='o')
# plt.plot(x, average_outside_temp,marker='o')
# plt.plot(x, average_hour,marker='o')
# plt.plot(x, average_outside_humidity,marker='o')

# plt.xlabel('Clusters', fontsize=14)
# plt.ylabel('Units of Parameters', fontsize=14)

# plt.legend(['average_lights', 'average_outside_temp', 'average_hour', 'average_outside_humidity'], loc='best')

# plt.show()


gmm_rudimentary_variation_score=stdev(sizes)
print('size variation score')
gmm_rudimentary_variation_score


# In[32]:


# Section 4
#Principal component analysis - both kmeans and gmm

# Just for analysis
# 10 component pca
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X)
# 3 component pca
pca = PCA(n_components=3)
principalComponents3 = pca.fit_transform(X)

# 6 component pca
pca = PCA(n_components=6)
principalComponents6 = pca.fit_transform(X)



# fitting principle components then doing k means using 3 components
km = KMeans(n_clusters = 4, max_iter = 500)
km.fit(principalComponents3)
pca_clusters=km.labels_
pca_clusters_km_labels=km.labels_

km_centers=km.cluster_centers_
km_centers_pca=km_centers
energy_source_data_features['pca_clusters']=pca_clusters

principalDf = pd.DataFrame(data = principalComponents6, columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5','principal component 6'])
finalDf = pd.concat([principalDf, energy_source_data_features[['pca_clusters']]], axis = 1)


# visualizing inter cluster k means distance
visualizer = InterclusterDistance(km)
visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show() 

# fitting principle components with expectation maximization
# Gausian Mixture
gmm = GMM(n_components=4).fit(principalComponents6)
gmm_labels = gmm.predict(principalComponents6)
energy_source_data_features['clusters_kmeans_em']=gmm_labels
gmm_centers=gmm.means_

# inter cluster distance and DV score for pca followed by k means
dists = euclidean_distances(km_centers)
tri_dists = dists[np.triu_indices(4, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
print('average inter cluster distance')
print(avg_dist)
print('dv_score kmeans')
print(davies_bouldin_score(X, pca_clusters)) 


# inter cluster distance and DV score for pca followed by GMM

dists = euclidean_distances(gmm_centers)
tri_dists = dists[np.triu_indices(4, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
print('average inter cluster distance')
print(avg_dist)
print('dv_score gmm')
print(davies_bouldin_score(X, gmm_clusters)) 



# kmeans 2d plot - pca followed by k means
sizes=energy_source_data_features['pca_clusters'].value_counts()
plot_set=pd.DataFrame()
plot_set['x']=principalComponents2[:,0]
plot_set['y']=principalComponents2[:,1]
plot_set['label']=energy_source_data_features['pca_clusters']
x = plot_set['x']
y = plot_set['y']
cluster = plot_set['label']  # Labels of cluster 0 to 3

ax = sns.scatterplot(x="x", y="y", hue="label", style="label", data=plot_set)

sizes=energy_source_data_features['pca_clusters'].value_counts()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
clust = ['Cluster0', 'Cluster1', 'Cluster2', 'Cluster3']

plt.xlabel('Clusters', fontsize=14)
plt.ylabel('Size of Clusters', fontsize=14)

ax.bar(clust,sizes)
plt.show()


pca_rudimentary_variation_score=stdev(sizes)
print('size variation score')
pca_rudimentary_variation_score


# In[33]:


# Section 5
#Independent component analysis - both kmeans and gmm

transformer = FastICA(n_components=10,
         random_state=0)
X_transformed = transformer.fit_transform(X)
X_transformed.shape



# fitting independent components with k means
km = KMeans(n_clusters = 4, max_iter = 500)
km.fit(X_transformed)
ica_clusters=km.labels_
km_centers=km.cluster_centers_
energy_source_data_features['ica_clusters']=ica_clusters

# ica followed by k means intercluster distance
visualizer = InterclusterDistance(km)
visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show() 

# fitting independent components with expectation maximization

gmm = GMM(n_components=4).fit(X_transformed)
gmm_labels = gmm.predict(X_transformed)
energy_source_data_features['clusters_ica_em']=gmm_labels
gmm_centers=gmm.means_


# inter cluster distance and dv score ica followed by k means
dists = euclidean_distances(km_centers)
tri_dists = dists[np.triu_indices(4, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
print('average inter cluster distance')
print(avg_dist)
print('dv_score kmeans')
print(davies_bouldin_score(X, ica_clusters)) 

# inter cluster distance and dv score ica followed by gmm

dists = euclidean_distances(gmm_centers)
tri_dists = dists[np.triu_indices(4, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
print('average inter cluster distance')
print(avg_dist)
print('dv_score gmm')
print(davies_bouldin_score(X, gmm_labels)) 



# kmeans 2d plot 
# ica followed by k means 
# plot for cluster distribution of size
sizes=energy_source_data_features['ica_clusters'].value_counts()
plot_set=pd.DataFrame()
plot_set['x']=principalComponents2[:,0]
plot_set['y']=principalComponents2[:,1]
plot_set['label']=energy_source_data_features['ica_clusters']
x = plot_set['x']
y = plot_set['y']
cluster = plot_set['label']  # Labels of cluster 0 to 3

ax = sns.scatterplot(x="x", y="y", hue="label", style="label", data=plot_set)



sizes=energy_source_data_features['ica_clusters'].value_counts()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
clust = ['Cluster0', 'Cluster1', 'Cluster2', 'Cluster3']

plt.xlabel('Clusters', fontsize=14)
plt.ylabel('Size of Clusters', fontsize=14)

ax.bar(clust,sizes)
plt.show()



ica_rudimentary_variation_score=stdev(sizes)
print('size variation score')
ica_rudimentary_variation_score


# In[35]:


# Section 6
#Random component analysis - both kmeans and gmm


transformer = random_projection.SparseRandomProjection(n_components=10)
X_new = transformer.fit_transform(X)
X_new.shape



# fitting random components with k means
km = KMeans(n_clusters = 4, max_iter = 500)
km.fit(X_new)
rca_clusters=km.labels_
rca_centers=km.cluster_centers_
energy_source_data_features['rca_clusters']=rca_clusters


visualizer = InterclusterDistance(km)
visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show() 

# fitting random components with expectation maximization
# Gausian Mixture
gmm = GMM(n_components=4).fit(X_new)
gmm_labels = gmm.predict(X_transformed)
energy_source_data_features['clusters_rca_em']=gmm_labels
gmm_centers=gmm.means_


# get inter cluster distance and dv score, random components followed by k means
dists = euclidean_distances(rca_centers)
tri_dists = dists[np.triu_indices(4, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
print('average inter cluster distance')
print(avg_dist)
print('dv_score kmeans')
print(davies_bouldin_score(X, rca_clusters)) 


# get inter cluster distance and dv score, random components followed by GMM
dists = euclidean_distances(gmm_centers)
tri_dists = dists[np.triu_indices(4, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
print('average inter cluster distance')
print(avg_dist)
print('dv_score gmm')
print(davies_bouldin_score(X, gmm_clusters)) 



# kmeans 2d plot and value count
# for cluster sizes
sizes=energy_source_data_features['rca_clusters'].value_counts()
plot_set=pd.DataFrame()
plot_set['x']=principalComponents2[:,0]
plot_set['y']=principalComponents2[:,1]
plot_set['label']=energy_source_data_features['rca_clusters']
x = plot_set['x']
y = plot_set['y']
cluster = plot_set['label']  # Labels of cluster 0 to 3

ax = sns.scatterplot(x="x", y="y", hue="label", style="label", data=plot_set)


sizes=energy_source_data_features['rca_clusters'].value_counts()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
clust = ['Cluster0', 'Cluster1', 'Cluster2', 'Cluster3']

plt.xlabel('Clusters', fontsize=14)
plt.ylabel('Size of Clusters', fontsize=14)

ax.bar(clust,sizes)
plt.show()


rca_rudimentary_variation_score=stdev(sizes)
print('size variation score')
rca_rudimentary_variation_score


# In[36]:


# Section 7
# Forward stepwise selection

clf = LogisticRegression()

# Build step forward feature selection
sfs1 = sfs(clf,
           k_features=20,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)

# Perform SFFS
sfs1 = sfs1.fit(x_train, y_train)

sfs1.k_feature_names_

#feat_cols = list(sfs1.k_feature_idx_)
forward_selection_feature_names=list(sfs1.k_feature_names_)


# In[38]:



# step forward followed by k means
km = KMeans(n_clusters = 4, max_iter = 500)

km.fit(X[forward_selection_feature_names])
forward_selection_clusters=km.labels_
energy_source_data_features['fwd_selection_clusters']=forward_selection_clusters
kmeans_centers=km.cluster_centers_




# fitting fwd selection components with expectation maximization
# Gausian Mixture
gmm = GMM(n_components=4).fit(X[forward_selection_feature_names])
gmm_labels = gmm.predict(X[forward_selection_feature_names])
energy_source_data_features['clusters_fwd_selection_em']=gmm_labels
gmm_centers=gmm.means_


# viz plot # step forward followed by k means
visualizer = InterclusterDistance(km)
visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show() 


#inter cluster distance and DV score # step forward followed by k means
dists = euclidean_distances(kmeans_centers)
tri_dists = dists[np.triu_indices(4, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
print('average inter cluster distance')
print(avg_dist)
print('dv_score kmeans')
print(davies_bouldin_score(X, forward_selection_clusters)) 


#inter cluster distance and DV score # step forward followed by k means
dists = euclidean_distances(gmm_centers)
tri_dists = dists[np.triu_indices(4, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
print('average inter cluster distance')
print(avg_dist)
print('dv_score gmm')
print(davies_bouldin_score(X, gmm_clusters)) 



# kmeans 2d plot and value count
#energy_source_data_features.groupby('kmeans_clusters').count()

# then cluster size plot
sizes=energy_source_data_features['fwd_selection_clusters'].value_counts()
plot_set=pd.DataFrame()
plot_set['x']=principalComponents2[:,0]
plot_set['y']=principalComponents2[:,1]
plot_set['label']=energy_source_data_features['fwd_selection_clusters']
x = plot_set['x']
y = plot_set['y']
cluster = plot_set['label']  # Labels of cluster 0 to 3

ax = sns.scatterplot(x="x", y="y", hue="label", style="label", data=plot_set)




sizes=energy_source_data_features['fwd_selection_clusters'].value_counts()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
clust = ['Cluster0', 'Cluster1', 'Cluster2', 'Cluster3']

plt.xlabel('Clusters', fontsize=14)
plt.ylabel('Size of Clusters', fontsize=14)

ax.bar(clust,sizes)
plt.show()



stepwise_rudimentary_variation_score=stdev(sizes)
print('size variation score')
stepwise_rudimentary_variation_score


# In[57]:


# scenerio 1 neural nets

# Neural Nets, most optimized version post dimensionality reduction using principle components
model = Sequential()


X_train_out, X_test_out, Y_train_out, Y_test_out = train_test_split(principalComponents3,energy_source_data_features['state'],test_size=0.2,random_state=121)


x_train=X_train_out
x_test=X_test_out
y_train=Y_train_out
y_test=Y_test_out
input_dim=x_train.shape[1]


encoding_test_y = np_utils.to_categorical(y_test)
encoding_train_y = np_utils.to_categorical(y_train)
activation='softsign'
#model.add(Dropout(0.01, input_shape=(21,)))
model.add(Dense(80,input_dim=input_dim))
model.add(Dense(80,activation=activation))
model.add(Dense(80,activation=activation))
#model.add(Dense(80,activation=activation))

model.add(Dense(2,activation=activation))

sgd = optimizers.SGD(learning_rate=0.10)

# Compiling model
model.compile(loss='squared_hinge', optimizer=sgd, metrics=['accuracy'])

# Fitting the model
model.fit(x_train, encoding_train_y,epochs=5, batch_size=2)
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
         lw=lw,label="area under curve = %1.3f" % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()



# In[56]:


# Neural Nets append clustering column as predictor along with principal components
model = Sequential()


# Creating a new dataset with 3 principal components and one clustering output
workset=principalDf[['principal component 1','principal component 2','principal component 3']]
workset['em_clust']=pca_clusters


# fitting neural nets on the above datasts
X_train_out, X_test_out, Y_train_out, Y_test_out = train_test_split(workset,energy_source_data_features['state'],test_size=0.2,random_state=121)


x_train=X_train_out
x_test=X_test_out
y_train=Y_train_out
y_test=Y_test_out
input_dim=x_train.shape[1]


encoding_test_y = np_utils.to_categorical(y_test)
encoding_train_y = np_utils.to_categorical(y_train)
activation='softsign'
#model.add(Dropout(0.01, input_shape=(21,)))
model.add(Dense(80,input_dim=input_dim))
model.add(Dense(80,activation=activation))
model.add(Dense(80,activation=activation))
#model.add(Dense(80,activation=activation))


model.add(Dense(2,activation=activation))

sgd = optimizers.SGD(learning_rate=0.10)

# Compiling model
model.compile(loss='squared_hinge', optimizer=sgd, metrics=['accuracy'])

# Fitting the model
model.fit(x_train, encoding_train_y,epochs=10, batch_size=3)
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
         lw=lw,label="area under curve = %1.3f" % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()



# In[60]:


model = Sequential()


# Creating new dataset using only clustering outputs as features
workset=pd.DataFrame()
workset['kmeans_clust']=kmeans_clusters
workset['gaus_label']=gmm_clusters
workset['pca_clusters']=pca_clusters
workset['ica_clusters']=ica_clusters
workset['rca_clusters']=rca_clusters


X_train_out, X_test_out, Y_train_out, Y_test_out = train_test_split(workset,energy_source_data_features['state'],test_size=0.2,random_state=121)


x_train=X_train_out
x_test=X_test_out
y_train=Y_train_out
y_test=Y_test_out

input_dim=x_train.shape[1]

# running neural nets on the above dataset
encoding_test_y = np_utils.to_categorical(y_test)
encoding_train_y = np_utils.to_categorical(y_train)
activation='softsign'
#model.add(Dropout(0.01, input_shape=(21,)))
model.add(Dense(80,input_dim=input_dim))
model.add(Dense(80,activation=activation))
model.add(Dense(80,activation=activation))
#model.add(Dense(80,activation=activation))


model.add(Dense(2,activation=activation))

sgd = optimizers.SGD(learning_rate=0.10)

# Compiling model
model.compile(loss='squared_hinge', optimizer=sgd, metrics=['accuracy'])

# Fitting the model
model.fit(x_train, encoding_train_y,epochs=5, batch_size=4)
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
         lw=lw,label="area under curve = %1.3f" % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()



# In[62]:


energy_source_data_features["state"] = energy_source_data_features["state"].astype('int')
energy_source_data_features.groupby(['pca_clusters'])['state'].agg('sum')

result = pd.concat([energy_source_data_features.groupby(['kmeans_clusters'])['state'].agg('sum'),energy_source_data_features.groupby(['rca_clusters'])['state'].agg('sum'),energy_source_data_features.groupby(['ica_clusters'])['state'].agg('sum'),energy_source_data_features.groupby(['pca_clusters'])['state'].agg('sum'), energy_source_data_features.groupby(['fwd_selection_clusters'])['state'].agg('sum')], axis=1, sort=False)
result.columns=['kmeans_original', 'rca_kmeans', 'ica_kmeans', 'pca_kmeans',
       'fwd_kmeans']

n=2298
result['kmeans_original']=result['kmeans_original']/n
result['rca_kmeans']=result['rca_kmeans']/n
result['ica_kmeans']=result['ica_kmeans']/n
result['pca_kmeans']=result['pca_kmeans']/n
result['fwd_kmeans']=result['fwd_kmeans']/n

x = np.arange(4)

# plot multiple line charts to show variation

plt.plot(x, result['kmeans_original'],marker='o')
plt.plot(x, result['rca_kmeans'],marker='o')
plt.plot(x, result['ica_kmeans'],marker='o')
plt.plot(x, result['pca_kmeans'],marker='o')
plt.plot(x, result['fwd_kmeans'],marker='o')

#plt.plot(x, average_outside_humidity,marker='o')

plt.xlabel('Clusters', fontsize=14)
plt.ylabel('Spike Dependent Var Breakdown', fontsize=14)

plt.legend(['Kmeans_original', 'rca_kmeans', 'ica_kmeans', 'pca_kmeans','fwd_kmeans'], loc='center')

plt.show()

kmeans_rudimentary_variation_score=stdev(sizes)
print('size variation score')
kmeans_rudimentary_variation_score


# In[63]:


dv_score=np.array(['1.059','1.06','1.07','1.2','2.24'])
plt.plot(dv_score,marker='o')
plt.xlabel('Techniques', fontsize=14)
plt.ylabel('DV Score', fontsize=14)
plt.show()

