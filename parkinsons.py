#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np #Mathematical calculation
import pandas as pd #data preprocessing
from sklearn.model_selection import train_test_split #split a dataset into two subsets
from sklearn.preprocessing import StandardScaler #transform data by standardizing it
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[3]:


parkinsons_data = pd.read_csv('parkinsons.csv') #take the data


# ##### parkinsons_data.head() #display first row
#  

# In[5]:


#number of rows and colums present

parkinsons_data.shape


# In[6]:


#information of the data set
parkinsons_data.info()


# In[7]:


parkinsons_data.isnull().sum()


# In[8]:


#satistical data 
parkinsons_data.describe() # generate descriptive statistics of a DataFrame


# In[9]:


#distribution telling how many people have parkinsons and how many doesnt
parkinsons_data['status'].value_counts()


# In[10]:


# 1 - parkinson
#0 - no


# In[11]:


#group bsed on the target
parkinsons_data.groupby('status').mean()


# In[12]:


#separate the target and the features
#column drop = axis =1 
#column drop axis = 0
x = parkinsons_data.drop(columns=['name', 'status'], axis=1)
y = parkinsons_data['status']


# In[13]:


print(x)


# In[14]:


print(y)


# In[15]:


#splitting - train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 2)


# In[16]:


print(x.shape, x_train.shape, x_test.shape)


# In[17]:


#data standardization
scaler = StandardScaler()


# In[18]:


#fit = fit all the data
#helps in understand data
# trains and comes to a common value for the data
scaler.fit(x_train)


# In[19]:


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[20]:


print(x_train)


# In[21]:


#model training
#support vector machine model
model = svm.SVC(kernel = 'linear')


# In[22]:


#training svm model with training data
model.fit(x_train, y_train)


# In[23]:


#acuracy store
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)


# In[24]:


print('acuracy:' ,training_data_accuracy)


# In[25]:


x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)


# In[26]:


print('acuracy:' ,test_data_accuracy)


# In[29]:


#prediction system

input_data = (153.04600,175.82900,68.62300,0.00742,0.00005,0.00364,0.00432,0.01092,0.05517,0.54200,0.02471,0.03572,0.05767,0.07413,0.03160,17.28000,0.665318,0.719467,-3.949079,0.357870,3.109010,0.377429)
#change data into numpy array

input_data_as_numpy_array = np.asarray(input_data)
#reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)

if(prediction[0]==0):
    print("the person has parkinsons")
else:
    print("the person doesnt have parkinsons")


# In[ ]:





# In[ ]:




