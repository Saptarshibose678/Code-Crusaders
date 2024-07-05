#!/usr/bin/env python
# coding: utf-8

# ### Import Data

# In[90]:


import pandas as pd
import numpy as np
import matplotlib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KDTree
from sklearn.ensemble import RandomForestClassifier


# ### Train Data

# In[10]:


df_wine = pd.read_csv('wine_quality_data/train', delim_whitespace=True)


# In[11]:


df_wine.head()


# In[12]:


df_wine.info()


# In[13]:


df_wine.isnull().sum()


# In[14]:


df_wine.describe()


# In[15]:


X_train = df_wine.iloc[:, :-1]
y_train = df_wine.iloc[:, -1]


# ### Test Data

# In[18]:


df_wine_test = pd.read_csv('wine_quality_data/test-sample', delim_whitespace=True)


# In[19]:


df_wine_test.head()


# In[20]:


df_wine_test.info()


# In[21]:


df_wine_test.isnull().sum()


# ### Visualization

# In[98]:


quality_check = df_wine.groupby('quality')['pH'].sum().reset_index()
quality_check


# In[99]:


graph1 = px.pie(quality_check, values='pH', names='quality', title='Total pH value as per quality of the wine')
graph1.show()


# In[100]:


quality_check1 = df_wine.groupby('quality')['f_acid'].sum().reset_index()
quality_check1


# In[101]:


graph2 = px.bar(quality_check1, x='quality', y='f_acid', title='Total f_acid value as per quality of the wine')
graph2.show()


# In[102]:


quality_check2 = df_wine.groupby('quality')['alcohol'].sum().reset_index()
quality_check2


# In[103]:


graph3 = px.pie(quality_check2, values='alcohol', names='quality', hole=0.6,
             title='Total alcohol value as per quality of the wine')
graph3.show()


# ### KD-Forest

# In[83]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[84]:


rf_model.fit(X_train, y_train)


# In[87]:


predicted_quality_random_forest = rf_model.predict(X_test)


# In[88]:


new_df1 = pd.concat([df_wine_test, pd.DataFrame(predicted_quality_random_forest, columns=['predicted_quality'])], axis=1)


# In[89]:


print("DataFrame with test data and predicted wine quality:")
new_df1.head()

