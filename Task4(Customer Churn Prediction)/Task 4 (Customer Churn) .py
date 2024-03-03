#!/usr/bin/env python
# coding: utf-8

# # SlashMark Internship Task 4
# ## Customer Churn Prediction

# **Problem Statement:** Use a dataset that includes customer demographics, usage data, and whether the customer churned or not. Employ machine learning techniques to build a predictive model that can identify customers likely to churn. 

# In[1]:


#import platform
import pandas as pd
import sklearn
import numpy as np
#import graphviz
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()


# ### Exploratory Data Analysis

# In[4]:


df.shape


# In[5]:


df.tail()


# In[6]:


df.size


# In[7]:


df.dtypes


# In[8]:


df.columns


# In[9]:


df.info()


# In[11]:


df.isnull().sum().sum()


# In[12]:


df.duplicated().sum()


# ### Basic Data Cleaning

# In[13]:


df['TotalCharges'].dtype


# In[14]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')


# In[15]:


df['TotalCharges'].dtype


# In[16]:


categorical_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
target = "Churn"


# In[17]:


df.skew(numeric_only = True)


# In[18]:


df.corr(numeric_only = True)


# ### Feature Distribution

# #### Numerical features distribution

# In[19]:


df[numerical_features].describe()


# In[20]:


df[numerical_features].hist(bins=30, figsize=(10,7))


# In[21]:


fig, ax = plt.subplots(1, 3, figsize=(14, 4))
df[df.Churn == "No"][numerical_features].hist(bins=30, color="blue", alpha=0.5, ax=ax)
df[df.Churn == "Yes"][numerical_features].hist(bins=30, color="red", alpha=0.5, ax=ax)


# #### Categorical feature distribution

# In[22]:


ROWS, COLS = 4, 4
fig, ax = plt.subplots(ROWS,COLS, figsize=(19,19))
row, col = 0, 0,
for i, categorical_feature in enumerate(categorical_features):
    if col == COLS - 1:
        row += 1
    col = i % COLS
    df[categorical_feature].value_counts().plot(kind='bar', ax=ax[row, col]).set_title(categorical_feature)


# In[23]:


feature = 'Contract'
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df[df.Churn == "No"][feature].value_counts().plot(kind='bar', ax=ax[0]).set_title('not churned')
df[df.Churn == "Yes"][feature].value_counts().plot(kind='bar', ax=ax[1]).set_title('churned')


# ### Target variable distribution

# In[24]:


df[target].value_counts().plot(kind='bar').set_title('churned')


# ### Outliers Analysis with IQR Method

# In[25]:


x = ['tenure','MonthlyCharges']
def count_outliers(data,col):
        q1 = data[col].quantile(0.25,interpolation='nearest')
        q2 = data[col].quantile(0.5,interpolation='nearest')
        q3 = data[col].quantile(0.75,interpolation='nearest')
        q4 = data[col].quantile(1,interpolation='nearest')
        IQR = q3 -q1
        global LLP
        global ULP
        LLP = q1 - 1.5*IQR
        ULP = q3 + 1.5*IQR
        if data[col].min() > LLP and data[col].max() < ULP:
            print("No outliers in",i)
        else:
            print("There are outliers in",i)
            x = data[data[col]<LLP][col].size
            y = data[data[col]>ULP][col].size
            a.append(i)
            print('Count of outliers are:',x+y)
global a
a = []
for i in x:
    count_outliers(df,i)


# ### Cleaning and Transforming Data

# No need of customerID 

# In[26]:


df.drop(['customerID'] , axis=1, inplace = True)


# In[27]:


df.head()


# #### On Hot Encoding

# In[28]:


df1=pd.get_dummies(data=df,columns=['gender', 'Partner', 'Dependents', 
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'], drop_first=True)


# In[29]:


df1.head()


# In[30]:


df1.columns


# #### Rearranging Columns

# In[31]:


df1 = df1[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'gender_Male', 'Partner_Yes', 'Dependents_Yes',
       'PhoneService_Yes', 'MultipleLines_No phone service',
       'MultipleLines_Yes', 'InternetService_Fiber optic',
       'InternetService_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No internet service',
       'OnlineBackup_Yes', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No internet service',
       'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No internet service', 'StreamingMovies_Yes',
       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check','Churn_Yes']]


# In[32]:


df1.head()


# In[33]:


df1.shape


# In[34]:


from sklearn.impute import SimpleImputer

# The imputer will replace missing values with the mean of the non-missing values for the respective columns

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

df1.TotalCharges = imputer.fit_transform(df1["TotalCharges"].values.reshape(-1, 1))


# #### Feature Scaling

# In[36]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[37]:


scaler.fit(df1.drop(['Churn_Yes'],axis = 1))
scaled_features = scaler.transform(df1.drop('Churn_Yes',axis = 1))


# #### Feature Selection

# In[38]:


from sklearn.model_selection import train_test_split
X = scaled_features
Y = df1['Churn_Yes']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 44)


# ### Prediction using Logistic Regression

# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score ,confusion_matrix

logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)


# In[40]:


predLR = logmodel.predict(X_test)


# In[41]:


predLR


# In[42]:


Y_test


# In[44]:


print(classification_report(Y_test, predLR))


# In[45]:


# calculate the classification report
report = classification_report(Y_test, predLR, target_names=['Churn_No', 'Churn_Yes'])

# split the report into lines
lines = report.split('\n')

# split each line into parts
parts = [line.split() for line in lines[2:-5]]

# extract the metrics for each class
class_metrics = dict()
for part in parts:
    class_metrics[part[0]] = {'precision': float(part[1]), 'recall': float(part[2]), 'f1-score': float(part[3]), 'support': int(part[4])}

# create a bar chart for each metric
fig, ax = plt.subplots(1, 4, figsize=(12, 4))
metrics = ['precision', 'recall', 'f1-score', 'support']
for i, metric in enumerate(metrics):
    ax[i].bar(class_metrics.keys(), [class_metrics[key][metric] for key in class_metrics.keys()])
    ax[i].set_title(metric)

# display the plot
plt.show()


# In[46]:


confusion_matrix_LR = confusion_matrix(Y_test, predLR)


# In[47]:


# create a heatmap of the matrix using matshow()

plt.matshow(confusion_matrix(Y_test, predLR))

# add labels for the x and y axes
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix_LR[i, j], ha='center', va='center')


# Add custom labels for x and y ticks
plt.xticks([0, 1], ["Not Churned", "Churned"])
plt.yticks([0, 1], ["Not Churned", "Churned"])
plt.show()


# In[48]:


logmodel.score(X_train, Y_train)


# In[49]:


accuracy_score(Y_test, predLR)


# ### Prediction using Support Vector Classifier

# In[50]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, Y_train)
y_pred_svc = svc.predict(X_test)


# In[51]:


print(classification_report(Y_test, y_pred_svc))


# In[52]:


confusion_matrix_svc = confusion_matrix(Y_test, y_pred_svc)


# In[53]:


# create a heatmap of the matrix using matshow()

plt.matshow(confusion_matrix_svc)

# add labels for the x and y axes
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix_svc[i, j], ha='center', va='center')

        
# Add custom labels for x and y ticks
plt.xticks([0, 1], ["Not Churned", "Churned"])
plt.yticks([0, 1], ["Not Churned", "Churned"])
plt.show()


# In[54]:


svc.score(X_train,Y_train)


# In[55]:


accuracy_score(Y_test, y_pred_svc)


# ### Prediction using Decision Tree Classifier

# In[56]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train, Y_train)
y_pred_dtc = dtc.predict(X_test)


# In[57]:


print(classification_report(Y_test, y_pred_dtc))


# In[58]:


confusion_matrix_dtc = confusion_matrix(Y_test, y_pred_dtc)


# In[59]:


# create a heatmap of the matrix using matshow()

plt.matshow(confusion_matrix_dtc)

# add labels for the x and y axes
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix_dtc[i, j], ha='center', va='center')


# Add custom labels for x and y ticks
plt.xticks([0, 1], ["Not Churned", "Churned"])
plt.yticks([0, 1], ["Not Churned", "Churned"])
plt.show()


# In[60]:


dtc.score(X_train, Y_train)


# In[61]:


accuracy_score(Y_test, y_pred_dtc)


# ### Prediction using KNN Classifier 

# In[62]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 30)
knn.fit(X_train, Y_train)


# In[63]:


pred_knn = knn.predict(X_test)


# In[64]:


error_rate= []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,Y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != Y_test))


# In[65]:


plt.figure(figsize = (10,6))
plt.plot(range(1,40),error_rate,color = 'blue',linestyle = '--',marker = 'o',markerfacecolor='red',markersize = 10)
plt.title('Error Rate vs K')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[66]:


print(classification_report(Y_test, pred_knn))


# In[67]:


confusion_matrix_knn = confusion_matrix(Y_test, pred_knn)


# In[68]:


# create a heatmap of the matrix using matshow()

plt.matshow(confusion_matrix_knn)

# add labels for the x and y axes
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix_knn[i, j], ha='center', va='center')

# Add custom labels for x and y ticks
plt.xticks([0, 1], ["Not Churned", "Churned"])
plt.yticks([0, 1], ["Not Churned", "Churned"])
plt.show()


# In[69]:


knn.score(X_train, Y_train)


# In[70]:


accuracy_score(Y_test, pred_knn)


# In[ ]:




