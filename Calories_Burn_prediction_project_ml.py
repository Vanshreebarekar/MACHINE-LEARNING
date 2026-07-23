#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# In[4]:


calories=pd.read_csv('calories.csv')


# In[5]:


calories.head()


# In[6]:


exercise_data=pd.read_csv('exercise.csv')


# In[7]:


exercise_data.head()


# In[8]:


calories_data=pd.concat([exercise_data,calories['Calories']],axis=1)


# In[9]:


calories_data.head()


# In[10]:


calories_data.shape


# In[11]:


calories_data.info()


# In[12]:


calories_data.isnull().sum()


# In[13]:


calories_data.describe()


# In[14]:


sns.set()


# In[17]:


sns.countplot(x='Gender', data=calories_data)


# In[18]:


sns.distplot(calories_data['Age'])


# In[19]:


sns.distplot(calories_data['Height'])


# In[20]:


sns.distplot(calories_data['Weight'])


# In[22]:


correlation = calories_data.select_dtypes(include=['number']).corr()


# In[23]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# In[24]:


calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)


# In[25]:


calories_data.head()


# In[26]:


X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']


# In[28]:


print(X)


# In[30]:


print(Y)


# In[31]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[32]:


print(X.shape, X_train.shape, X_test.shape)


# # XGBoost Regressor

# In[33]:


model = XGBRegressor()


# In[34]:


model.fit(X_train, Y_train)


# In[36]:


data_prediction = model.predict(X_test)


# In[37]:


print(data_prediction)


# In[39]:


mae = metrics.mean_absolute_error(Y_test, data_prediction)


# In[42]:


mae # mean absolute error


# In[47]:


r2_score = metrics.r2_score(Y_test,data_prediction)
print("R-squared score = ", r2_score)


# # model is performing extremely well on the given dataset!

# In[ ]:




