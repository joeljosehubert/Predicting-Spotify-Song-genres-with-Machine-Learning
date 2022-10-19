#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[3]:


#Load the training data
train = pd.read_csv('/Users/noeljoseph/Downloads/Coding/Spotify Classification/Trainset.csv')
train.head(7)


# In[4]:


print(train.columns.values)


# In[5]:


#Drop unwanted columns
train = train.drop(['title', 'artist', 'year'], axis=1)


# In[6]:


train.isna().sum()


# In[7]:


train = train.dropna()


# In[9]:


#Get a count of the number of various top genre of songs
train['top genre'].value_counts()


# In[10]:


#Visualise the count
plt.figure(figsize=(30, 6))
sns.countplot(train['top genre'], label = 'Count')


# In[11]:


#Check for columns that need to be encoded
train.dtypes


# In[12]:


#Encode the categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
train.iloc[:,-1] = labelencoder_Y.fit_transform(train.iloc[:,-1].values)


# In[15]:


#Create a pair plot
sns.pairplot(train.iloc[:,4:12], hue='top genre')


# In[16]:


#Print the first seven rows of the new data
train.head(7)


# In[17]:


#Remove Id from the dataset
train = train.drop('Id', axis=1)


# In[18]:


#Generate the correlation of the columns
train.iloc[:,:11].corr()


# In[19]:


#Visualise the correlation of the columns
plt.figure(figsize=(20, 10))
sns.heatmap(train.iloc[:,:11].corr(), annot=True, fmt='.0%')


# In[21]:


#Load the testing data
test = pd.read_csv('/Users/noeljoseph/Downloads/Coding/Spotify Classification/Testset.csv')
test.head(7)


# In[22]:


#Drop unwanted columns
test = test.drop(['title', 'artist', 'year'], axis=1)


# In[23]:


test.shape


# In[24]:


#Remove Id from the test dataset
test = test.drop('Id', axis=1)


# In[26]:


#Now split the training data into independent (X) and dependent (Y) datasets
X_train = train.drop('top genre', axis=1)
Y_train = train['top genre']
X_test  = test
X_train.shape, Y_train.shape, X_test.shape


# In[27]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
result_log = round(logreg.score(X_train, Y_train) * 100, 2)
result_log


# In[28]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
result_knn = round(knn.score(X_train, Y_train) * 100, 2)
result_knn


# In[29]:


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
result_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
result_gaussian


# In[30]:


# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
result_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
result_perceptron


# In[31]:


# Linear SVC

l_svc = LinearSVC()
l_svc.fit(X_train, Y_train)
Y_pred = l_svc.predict(X_test)
result_linear_svc = round(l_svc.score(X_train, Y_train) * 100, 2)
result_linear_svc


# In[32]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
result_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
result_sgd


# In[33]:


# Decision Tree
d_tree = DecisionTreeClassifier()
d_tree.fit(X_train, Y_train)
Y_pred = d_tree.predict(X_test)
result_decision_tree = round(d_tree.score(X_train, Y_train) * 100, 2)
result_decision_tree


# In[34]:


random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
result_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
result_random_forest


# In[37]:


prediction = random_forest.predict(X_test)


# In[38]:


print(prediction)

