#!/usr/bin/env python
# coding: utf-8

# # SlashMark Internship Task 1
# ## Analyze Daily Weather Data
# 
# **Problem Statement:** Use a small dataset of daily weather information (temperature, precipitation, etc.) and Analyze Weather on Daily basis.

# ## Data Exploration

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[3]:


df=pd.read_csv("weather.csv")


# In[4]:


print(df)


# In[5]:


df.head(10)


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# # Data Visualization - I

# In[9]:


sns.pairplot(df[['MinTemp','MaxTemp','Evaporation','Sunshine','RISK_MM']])
plt.show()


# # Data Analysis: Calculate statistics or relationships for MinTemp

# In[10]:


print("Mean of MinTemp ",df["MinTemp"].mean())


# In[11]:


print("Median of MinTemp ",df["MinTemp"].median())


# In[12]:


print("STD of MinTemp ",df["MinTemp"].std())


# # Data Analysis: Calculate statistics or relationships for Rainfall

# In[13]:


print("Mean of Rainfall ",df["Rainfall"].mean())


# In[14]:


print("Median of Rainfall ",df["Rainfall"].median())


# In[15]:


print("STD of Rainfall ",df["Rainfall"].std())


# # correleation between two columns

# In[16]:


df['MinTemp'].corr(df['MaxTemp'])


# In[17]:


plt.scatter(df['MinTemp'],df['MaxTemp'])
plt.xlabel("MinTemp")
plt.ylabel("MaxTemp")
plt.title("Scatter Plot")
plt.show()


# In[18]:


plt.scatter(df["Rainfall"],df["Evaporation"])
plt.xlabel("Rainfall")
plt.ylabel("Evaporation")
plt.title("Scatter Plot")
plt.show()


# In[19]:


plt.hist(df["MinTemp"], bins=10, edgecolor='black')
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Histogram")
plt.show()


# In[20]:


plt.boxplot(df["MinTemp"])
plt.xlabel("MinTemp")
plt.ylabel("Values")
plt.title("Box Plot")
plt.show()


# In[21]:


plt.boxplot(df["Evaporation"])
plt.xlabel("Evaporation")
plt.ylabel("Values")
plt.title("Box Plot")
plt.show()


# # Prediction

# In[22]:


X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[24]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[25]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse}')


# In[ ]:




