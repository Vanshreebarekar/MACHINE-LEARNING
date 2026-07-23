#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_style
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('Iris (1).csv')
df.head()


# In[3]:


# delete a column
df = df.drop(columns = ['Id'])
df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df['Species'].value_counts()


# # PREPROCESSING DATASETS

# In[7]:


df.isnull().sum()


# In[8]:


# Exploratry data anatlsis


# In[9]:


df['SepalLengthCm'].hist()


# In[10]:


df['SepalWidthCm'].hist()


# In[11]:


df['PetalLengthCm'].hist()


# In[12]:


df['PetalWidthCm'].hist()


# In[13]:


colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[14]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[15]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[16]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[17]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# In[18]:


df.corr(numeric_only=True)


# In[19]:


# 1. Calculate the numbers (add parentheses and numeric_only)
corr = df.corr(numeric_only=True)

# 2. Create the plot
fig, ax = plt.subplots(figsize=(5, 4))

# 3. Use capital 'True' for annot
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')


# # Label Encoder

# In[20]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[21]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# In[22]:


from sklearn.model_selection import train_test_split
#train - 70
#test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[23]:


# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[24]:


model.fit(x_train, y_train)


# In[25]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[27]:


model.fit(x_train, y_train)


# In[28]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[30]:


model.fit(x_train, y_train)


# In[31]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[32]:


import pickle
filename='model.pkl'
pickle.dump(model,open(filename,'wb'))


# In[33]:


x_test.head()


# In[34]:


load_model = pickle.load(open('model.pkl','rb'))


# In[35]:


import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


iris = load_iris()
X = iris.data
y = iris.target


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = RandomForestClassifier(n_estimators=100)
model.fit(X_scaled, y)

# 4. Save the Mode
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 5. Save the Scaler (Your app.py needs this to "translate" user input)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Proper model and scaler saved successfully!")


# In[ ]:




