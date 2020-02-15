import praw
import pandas as pd
import datetime as dt
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
#import logging
import numpy as np
from numpy import random
from sklearn.linear_model import SGDClassifier


reddit = praw.Reddit(client_id='anw_EpVABeFTzg', client_secret='CuEWY3Fmt1CIE1lrI12ypbfjGrc', user_agent='flair_detector', username='dev_sid', password='hitechcamp')
subreddit = reddit.subreddit('india')
flairs = ["AskIndia", "Non-Political", "[R]eddiquette", "Scheduled", "Photography", "Science/Technology", "Politics", "Business/Finance", "Policy/Economy", "Sports", "Food", "AMA"]
topics_dict = {"flair":[], "title":[], "score":[], "id":[], "url":[], "comms_num": [], "body":[], "author":[], "comments":[]}
k=0
#for flair in flairs:
  
 # get_subreddits = subreddit.search(flair, limit=50)
  
  #for submission in get_subreddits:
    
   # topics_dict["flair"].append(flair)
    #topics_dict["title"].append(submission.title)
    #topics_dict["score"].append(submission.score)
    #topics_dict["id"].append(submission.id)
    #topics_dict["url"].append(submission.url)
    #topics_dict["comms_num"].append(submission.num_comments)
    #topics_dict["body"].append(submission.selftext)
    #topics_dict["author"].append(submission.author)
    
    #submission.comments.replace_more(limit=None)
    #comment = ''
    #for top_level_comment in submission.comments:
     # comment = comment + ' ' + top_level_comment.body
    #topics_dict["comments"].append(comment)
    #k+=1
    #print(k)
    
#df = pd.DataFrame(topics_dict)
#df.to_csv('data.csv', index=False) 

df = pd.read_csv('data.csv')
#df.head(10)


df['body'] = df['body'].astype(str)
df['title'] = df['title'].astype(str)
df['comments'] = df['comments'].astype(str)
df['url'] = df['url'].astype(str)
stemmer = PorterStemmer()
words = stopwords.words("english")

df['title'] = df['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
df['body'] = df['body'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
df['comments'] = df['comments'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
df['url'] = df['url'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
df['title'] = df['title'] + ' ' + df['body']


def nb_classifier(X_train, X_test, y_train, y_test):
  
  from sklearn.naive_bayes import MultinomialNB


  nb = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('clf', MultinomialNB()),
                ])
  nb.fit(X_train, y_train)

  y_pred = nb.predict(X_test)

  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred,target_names=flairs))

def linear_svm(X_train, X_test, y_train, y_test):
  
  from sklearn.linear_model import SGDClassifier

  sgd = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                 ])
  sgd.fit(X_train, y_train)

  y_pred = sgd.predict(X_test)

  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred,target_names=flairs))

def logisticreg(X_train, X_test, y_train, y_test):

  from sklearn.linear_model import LogisticRegression

  logreg = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                 ])
  logreg.fit(X_train, y_train)

  y_pred = logreg.predict(X_test)

  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred,target_names=flairs))


def randomforest(X_train, X_test, y_train, y_test):
  
  from sklearn.ensemble import RandomForestClassifier
  
  ranfor = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('clf', RandomForestClassifier(n_estimators = 1000, random_state = 42)),
                 ])
  ranfor.fit(X_train, y_train)

  y_pred = ranfor.predict(X_test)

  print('accuracy %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred,target_names=flairs))

def train_test(X,y):
 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

  print("Results of Naive Bayes Classifier")
  nb_classifier(X_train, X_test, y_train, y_test)
  print("Results of Linear Support Vector Machine")
  linear_svm(X_train, X_test, y_train, y_test)
  print("Results of Logistic Regression")
  logisticreg(X_train, X_test, y_train, y_test)
  print("Results of Random Forest")
  randomforest(X_train, X_test, y_train, y_test)

t = df.flair

a = df.title
b = df.comments
c = df.body
d = df.url

print("Flair Detection using Title as Feature")
train_test(a,t)
print("Flair Detection using Body as Feature")
train_test(c,t)
print("Flair Detection using URL as Feature")
train_test(d,t)
print("Flair Detection using Comments as Feature")
train_test(b,t)

X_train, X_test, y_train, y_test = train_test_split(a, t, test_size=0.3, random_state = 42)
#model = SGDClassifier()
model = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                 ])
model.fit(X_train, y_train)

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))