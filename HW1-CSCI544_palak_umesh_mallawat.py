#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Palak Umesh Mallawat
#Python 3
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
import contractions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# ## Read Data

# In[8]:


df=pd.read_table('data.tsv', on_bad_lines = 'skip')


# ## Keep Reviews and Ratings

# In[9]:


df['review_body']=df['review_body'].apply(str) 
dff = df.loc[:, ['star_rating','review_body']]


#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# In[10]:


df1=dff.loc[df['star_rating'].isin(['1','2'])]
df1['star_rating']=1
df2=dff.loc[df['star_rating'] == '3']
df2['star_rating']=2
df3=dff.loc[df['star_rating'].isin(['4','5'])]
df3['star_rating']=3
df1=df1.sample(n=20000)
df2=df2.sample(n=20000)
df3=df3.sample(n=20000)
dfr=pd.concat([df1,df2,df3])


# # Data Cleaning
# 
# 

# # Pre-processing

# In[11]:


avgCharLengthBeforeCleaning=dfr['review_body'].str.len().mean()
dfr['review_body'] = dfr['review_body'].str.lower()
dfr['review_body']=dfr['review_body'].apply(str)
 # strip html with BeautifulSoup
dfr['review_body'] = [BeautifulSoup(text).get_text() for text in dfr['review_body'] ]
# remove non alphabetic. keep spaces
dfr['review_body'] = dfr['review_body'].str.replace('[^a-zA-Z ]', '')
# strip leading and trailing spaces. strip extra white spaces
dfr['review_body'] = dfr['review_body'].str.strip()
# handle contractions
dfr['review_body'] = [contractions.fix(text) for text in dfr['review_body'] ]
# get average length of reviews after cleaning
avgCharLengthAfterCleaning=dfr['review_body'].str.len().mean()
print("Printing the average character count before and after cleaning  "+ str(avgCharLengthBeforeCleaning) + ", " + str(avgCharLengthAfterCleaning))


# ## remove the stop words 

# In[12]:


stop = stopwords.words('english')
dfr['review_body'] = dfr['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# ## perform lemmatization  

# In[13]:


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

dfr['review_body'] = dfr['review_body'].apply(lemmatize_text)
dfr['review_body'] = dfr['review_body'].apply(str)
avgCharLengthAfterProcessing=dfr['review_body'].str.len().mean()
print("Printing the average character count before and after Preprocessing  "+ str(avgCharLengthAfterCleaning) + ", " + str(avgCharLengthAfterProcessing))


# # TF-IDF Feature Extraction

# In[14]:


v = TfidfVectorizer()
x = v.fit_transform(dfr['review_body'])
# x.shape


# # function for printing values in the required format

# In[60]:


def printValues(value):
    print(str(value['1']['precision']) + ", " + str(value['1']['recall']) + ", " + str(value['1']['f1-score']))
    print(str(value['2']['precision']) + ", " + str(value['2']['recall']) + ", " + str(value['2']['f1-score']))
    print(str(value['3']['precision']) + ", " + str(value['3']['recall']) + ", " + str(value['3']['f1-score']))
    print(str(value['macro avg']['precision']) + ", " + str(value['macro avg']['recall']) + ", " + str(value['macro avg']['f1-score']))


# # spliting the data into train and test 

# In[61]:


X_train, X_test, y_train, y_test = train_test_split(x, dfr['star_rating'], test_size = 0.2, random_state = 42)
y_train = y_train.astype('int')
y_test = y_test.astype('int')


# # Perceptron

# In[62]:


p = Perceptron(n_jobs = -1, max_iter = 10000, random_state = 42)
p.fit(X_train, y_train)
printValues(classification_report(p.predict(X_test), y_test, output_dict=True))


# # SVM

# In[63]:


sv = svm.SVC(kernel='linear')
sv.fit(X_train, y_train)
printValues(classification_report(sv.predict(X_test), y_test, output_dict=True))


# # Logistic Regression

# In[64]:


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
classification_report(lr.predict(X_test), y_test, output_dict=True)
printValues(classification_report(lr.predict(X_test), y_test, output_dict=True))


# # Naive Bayes

# In[65]:


mltnb = MultinomialNB()
mltnb.fit(X_train, y_train)
printValues(classification_report(mltnb.predict(X_test), y_test, output_dict=True))

