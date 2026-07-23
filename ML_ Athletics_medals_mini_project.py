#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style ('darkgrid')


# In[2]:


teams=pd.read_csv("teams.csv")


# In[3]:


teams


# In[9]:


teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]


# In[14]:


teams


# In[19]:


teams.corr(numeric_only=True)["medals"]


# In[20]:


sns.lmplot(x="athletes",y="medals",data=teams,fit_reg=True,ci=None)#ci=confidence interval.


# In[21]:


sns.lmplot(x='age', y='medals', data=teams, fit_reg=True, ci=None) 


# In[22]:


teams.plot.hist(y="medals")


# In[23]:


teams[teams.isnull().any(axis=1)].head(20)


# In[24]:


teams = teams.dropna()


# In[25]:


teams.shape


# In[26]:


train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()


# In[27]:


train.shape


# In[28]:


test.shape


# In[46]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()

predictors = ["athletes", "prev_medals"]

model.fit(train[predictors], train["medals"])




# In[47]:


predictions = model.predict(test[predictors])


# In[48]:


predictions.shape


# In[64]:


test["predictions"] = predictions


# In[65]:


test.loc[test["predictions"] < 0, "predictions"] = 0


# In[66]:


test["predictions"] = test["predictions"].round()


# In[67]:


test


# In[52]:


from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(test["medals"], test["predictions"])
error


# In[68]:


teams.describe()["medals"]


# In[54]:


test["predictions"] = predictions


# In[55]:


test[test["team"] == "USA"]


# In[56]:


test[test["team"] == "IND"]


# In[57]:


errors = (test["medals"] - predictions).abs()


# In[71]:


error_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio =  error_by_team / medals_by_team 


# In[72]:


error_ratio = error_ratio[np.isfinite(error_ratio)]

error_ratio
# In[61]:


error_ratio.plot.hist()


# In[62]:


error_ratio.sort_values() # {Error Ratio} =[{Actual Medals} - {Predicted Medals}]/{Actual Medals}


# In[ ]:




