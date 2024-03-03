#!/usr/bin/env python
# coding: utf-8

# ## SlashMark Internship Task 2
# ### Task 2: Fake News Detection
# #### Problem Statement

# ![Problem%20Statement.png](attachment:Problem%20Statement.png)

# ### About the Dataset 
# 1. *id:* unique id for a news article
# 2. *title:* the title of a news article 
# 3. *author:* author of the news article 
# 4. *text:* the text of the article; could be incomplete
# 5. *label:* a label that marks whether the news article is real or fake:
# 
# 1: Fake news\
# 0: Real news

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk


# In[2]:


nltk.download('stopwords')


# #### Printing the stopwords in English

# In[3]:


print(stopwords.words('english'))


# ### Data Pre-processing

# #### Loading the dataset to a pandas DataFrame

# In[4]:


news_data = pd.read_csv('train.csv')
news_data.head()


# In[5]:


news_data.shape


# #### Handling Missing Values

# In[6]:


news_data.isnull().sum()


# In[7]:


news_data = news_data.fillna('')
news_data.isnull().sum()


# #### Merging the author name and news title

# In[8]:


news_data['content'] = news_data['author']+' '+news_data['title']
print(news_data['content'])
news_data


# #### Separating the data & Label

# In[9]:


X = news_data.drop(columns='label',axis=1)
Y = news_data['label']
print(X)
print(Y)


# ### Stemming: 
# #### Stemming is the process of reducing a word to its Root word 
# #### example: actor, actress , acting -> act

# In[10]:


port_stem = PorterStemmer()


# In[11]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[12]:


news_data['content'] = news_data['content'].apply(stemming)


# In[13]:


print(news_data['content'])


# #### Seperate Data and Label

# In[14]:


X = news_data['content'].values
Y = news_data['label'].values


# In[15]:


print(X)


# In[16]:


print(Y)


# In[17]:


Y.shape


# #### Converting textual Data into Numerical Data

# In[18]:


vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[19]:


print(X)


# ### Splitting Dataset into Training and Test Data

# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)


# #### Training the Model : Logistic Regression
# 

# In[21]:


model = LogisticRegression()


# In[22]:


model.fit(X_train, Y_train)


# ### Evaluation

# #### Accuracy Score

# In[23]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[24]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[25]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[26]:


print('Accuracy score of the test data : ', test_data_accuracy)


# ### Making a Prediction System
# 

# In[31]:


X_new = X_test[1]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')


# In[32]:


print(Y_test[1])


# In[ ]:




