#!/usr/bin/env python
# coding: utf-8

# In[47]:


# Spam->"Bad mail"   &    Ham->"Good mail"


# In[48]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer# text->>numeric value
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[49]:


raw_mail_data = pd.read_csv('mail_data.csv')


# In[50]:


raw_mail_data 


# In[51]:


# replace the null values with a null string

mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[52]:


mail_data.head()


# In[53]:


mail_data.shape


# In[54]:


# label spam mail as 0;  ham mail as 1;

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# In[55]:


X = mail_data['Message']

Y = mail_data['Category']


# In[56]:


X


# In[57]:


Y


# In[58]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[59]:


X.shape


# In[60]:


X_train.shape


# In[61]:


X_test.shape


# In[62]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_test = encoder.transform(Y_test)


# In[63]:


# transform the text data to feature vectors that can be used as input to the Logistic regression

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[64]:


X_train


# In[65]:


X_train_features


# # Logistic Regression()

# In[66]:


model = LogisticRegression()


# In[67]:


model.fit(X_train_features, Y_train)


# In[68]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[69]:


accuracy_on_training_data #Accuracy is good


# In[70]:


prediction_on_training_data


# # Building a Predictive System>> Spam or Ham mail
# 

# In[71]:


input_mail = ["Congratulations! You won ₹50,000 cash prize. Click the link to claim now!"]
# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)


if (prediction[0]==0):
  print('Ham mail')

else:
  print('Spam mail')


# In[72]:


input_mail = ["Your order #45892 has been shipped and will arrive by Friday."]
# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)


if (prediction[0]==0):
  print('Ham mail')

else:
  print('Spam mail')


# In[ ]:





# In[ ]:





# In[ ]:




