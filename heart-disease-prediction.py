#!/usr/bin/env python
# coding: utf-8

# # HEART DISEASE RISK PREDICTION

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


df = pd.read_csv("heart_cleveland_upload.csv.csv")


# In[13]:


df.head()


# In[14]:


df.tail()


# In[19]:


df.info()


# In[20]:


df.isnull().sum()


# In[22]:


dp_val = df.duplicated().any()
print(dp_val)


# # Exploring Relationship : Heatmaps with Python for Data Visualization

# In[30]:


plt.figure(figsize=(12,7))
sns.heatmap(df.corr(), annot=True)


# # Number Affected And Unaffected by Heart Problems

# In[31]:


df.columns


# In[35]:


sns.countplot(x='condition',data=df)
plt.xticks(rotation=0)
plt.show()


# ### GENDER DISTRIBUTION IN DATASET

# In[36]:


df.columns


# In[39]:


sns.countplot(x='sex',data=df)
plt.xticks(rotation=0)
plt.xlabel("sex")
plt.ylabel("count")
plt.show()


# # Distribution of heart diseases among Males and Females

# In[40]:


df.columns


# In[44]:


sns.countplot(data=df, x=df['sex'], hue=df['condition'])
plt.xticks([0,1],['Female','Male'])
plt.legend(labels=['No-Disease', 'Disease'])
plt.show()


# # Distribution of Ages in Our Dataset

# In[45]:


df.columns


# In[49]:


sns.distplot(df['age'], bins=20)
plt.show()


# # Chest Pain Types

# In[62]:


sns.countplot(x='cp',hue='cp', data=df,palette='Set2',legend=False)
plt.xticks([0,1,2,3],["Typical angina"," Atypical angina"," Non-anginal pain","Asymptomatic"])
plt.xticks(rotation=0)
plt.xlabel("chest pain type")
plt.ylabel("count")
plt.show()


# # Chest Pain Distribution in Heart Disease VS Non-Disease

# In[63]:


sns.countplot(data=df, x='cp', hue='condition')
plt.legend(labels=["Not_Disease","Disease"])
plt.show()


# # Resting Blood Pressure Distribution Overview

# In[64]:


df.columns


# In[65]:


df['trestbps'].hist()


# # Serum Cholestrol(chol) Data Distribution

# In[66]:


df['chol'].hist()


# train test model
# 

# In[75]:


x = df.drop('condition',axis=1)
y= df['condition']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# feature scalling

# In[76]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# model training (LogisticRegression)

# In[77]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train_scaled, y_train)


# Prediction

# In[78]:


from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Confussion Matrix

# In[79]:


from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




