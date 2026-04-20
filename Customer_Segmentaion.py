#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pip', 'install yellowbrick')


# In[3]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


from yellowbrick.cluster import KElbowVisualizer


# In[4]:


df=pd.read_csv('customer_segmentation.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df['Income']=df['Income'].fillna(df['Income'].median())


# In[10]:


df['Dt_Customer']=pd.to_datetime(df['Dt_Customer'],dayfirst=True)


# In[11]:


df['Age']=2026-df['Year_Birth']


# In[12]:


spent_col = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

df['Total_Spending'] = df[spent_col].sum(axis = 1)


# In[13]:


df['Customer_Since'] = (pd.Timestamp('today') - df['Dt_Customer']).dt.days


# In[14]:


corr = df[['Income', 'Age', 'Recency', 'Total_Spending', 'NumWebPurchases', 'NumStorePurchases']]
corr


# In[15]:


sns.heatmap(corr.corr(),annot = True, cmap = 'coolwarm')
plt.title('Correlation Matrx')
plt.show()


# In[16]:


pivot_income = df.pivot_table(values = 'Income', index = 'Education', columns = 'Marital_Status', aggfunc = 'mean')
pivot_income


# In[17]:


group1 = df.groupby('Education')['Total_Spending'].mean().sort_values(ascending = False)


# In[18]:


group1


# In[19]:


group2=df.groupby('Marital_Status')['Total_Spending'].mean().sort_values(ascending=False)


# In[20]:


group2


# In[21]:


sns.catplot(group1, kind = 'bar', color = 'salmon', edgecolor = 'black')
plt.title('Average Spending by Education')
plt.ylabel('Average total Spending')
plt.xticks(rotation = 45)
plt.show()


# In[22]:


sns.catplot(group2, kind = 'bar', color = 'pink', edgecolor = 'black')
plt.title('Average Spending by Marital Status')
plt.ylabel('Average total Spending')
plt.xticks(rotation = 45)
plt.show()


# In[23]:


df['AcceptedAny'] = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']].sum(axis = 1)


# In[24]:


df['AcceptedAny'].unique()


# In[25]:


df['AcceptedAny']=df['AcceptedAny'].apply(lambda x:1 if x>0 else 0)


# In[26]:


df['AcceptedAny'].unique()


# In[27]:


group3=df.groupby('Marital_Status')['AcceptedAny'].mean().sort_values(ascending=False)


# In[28]:


sns.catplot(group3, kind = 'bar', color = 'orange', edgecolor = 'black')
plt.title('Average Acceptance by Marital Status')
plt.ylabel('Average Acceptance')
plt.xticks(rotation = 45)
plt.show()


# In[29]:


bins = [18, 30, 40, 50, 60, 70, 90]
labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']


df['AgeGroup'] = pd.cut(df['Age'], bins = bins, labels = labels)
df['AgeGroup']


# In[30]:


group4 = df.groupby('AgeGroup')['Income'].mean()
group4


# In[31]:


sns.catplot(group4, kind = 'bar', palette = 'RdYlGn', edgecolor = 'black')
plt.title('Average Income By Age')
plt.ylabel('Income')
plt.xticks(rotation = 45)
plt.show()


# In[32]:


features = ['Age', 'Income', 'Total_Spending', 'NumWebPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']


# In[33]:


X=df[features].copy()


# In[34]:


X


# In[35]:


scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)


# In[36]:


wcss = []
Silhouette_score = []


for i in range(2, 16):
    kmeans = KMeans(n_clusters = i, random_state = 42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    wcss.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, cluster_labels)

    Silhouette_score.append(sil_score)


# In[37]:


plt.figure(figsize = (12,6))
plt.subplot(121)
sns.lineplot(x = np.arange(2, 16), y = wcss, marker = 'o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('K - Value')
plt.ylabel('WCSS Value')
plt.grid(True)

plt.subplot(122)
sns.lineplot(x = np.arange(2,16), y = Silhouette_score, marker = 'o')
plt.title('Silhouette Score For Optimal K')
plt.xlabel('Number Of Clusters')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.grid(True)
plt.show()


# In[38]:


kmeans = KMeans(random_state = 42)
elbow = KElbowVisualizer(kmeans, k = (2, 16))
elbow.fit(X_scaled)
elbow.show(block = True)
elbow.elbow_value_


# In[39]:


kmeans = KMeans(n_clusters = 5, random_state = 42)
df['Cluster'] = kmeans.fit_predict(X_scaled)


# In[40]:


df.head()


# In[41]:


cluster_summary=df.groupby('Cluster')[features].mean()


# In[42]:


cluster_summary


# In[43]:


df['Cluster'].value_counts()


# In[44]:


pca = PCA(n_components = 2)
pca_data = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = pca_data[:, 0], pca_data[:, 1]


# In[45]:


pca_data


# In[46]:


sns.relplot(data = df, x = 'PCA1', y = 'PCA2', hue = 'Cluster', palette = 'deep')
plt.title('Customer Segmentation (PCA)')
plt.grid(False)
plt.show()


# In[ ]:


#-------------------------------------END----------------------------------------

