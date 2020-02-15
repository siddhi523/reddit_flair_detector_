import sklearn
import pickle
import praw
import re
from bs4 import BeautifulSoup
import nltk
#nltk.download('all')
from nltk.corpus import stopwords

reddit = praw.Reddit(client_id='anw_EpVABeFTzg', client_secret='CuEWY3Fmt1CIE1lrI12ypbfjGrc', user_agent='flair_detector', username='dev_sid', password='hitechcamp')
loaded_model = pickle.load(open('Model/finalized_model.sav', 'rb'))
def detect_flair(url,loaded_model):

  submission = reddit.submission(url=url)

  data = {}

  data['title'] = submission.title
  data['body'] = submission.selftext
  #data['body'] = data['body'].astype(str)
  #data['title'] = data['title'].astype(str)

  #data['title'] = data['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
  #data['body'] = data['body'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
  data['combine'] = data['title'] + data['body'] 
    
  return loaded_model.predict([data['combine']])
