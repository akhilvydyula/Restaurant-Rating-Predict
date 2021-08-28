import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib

import re
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"Rated", "", phrase)
    phrase = re.sub(r"RATED", "", phrase)
    phrase = re.sub(r"grt", "great", phrase)
    phrase = re.sub(r"v.good", "very good", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't","rated", "\n", "n","nan", "x"])
    

def clean_text(sentance):
    sentance = re.sub(r"http\S+", "", sentance)
   
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    return sentance.strip()

def predict(string):
    clf = joblib.load('model1.pkl')
    tfidf_vect = joblib.load('tfidf_vect.pkl')
    review_text = clean_text(string)
    test_vect = tfidf_vect.transform(([review_text]))
    pred = clf.predict(test_vect)
    
dataset=pd.read_csv("preprocessed_data.csv")
dataset=dataset.drop(["Unnamed: 0","online_order","book_table","votes","location","rest_type","cuisines","avg_2_ppl_cost","listed_type","cuisine_count"],axis=1)
dataset.dropna(inplace=True)

preprocessed_reviews = []
for sentence in dataset['reviews_list'].values:
    preprocessed_reviews.append(clean_text(sentence))
    
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(preprocessed_reviews)
joblib.dump(tfidf_vect, 'tfidf_vect.pkl')

X = count_vect.transform(preprocessed_reviews)

y = dataset['rate'].values

clf=RandomForestRegressor(n_estimators=15, random_state=100)
clf.fit(X,y)
joblib.dump(clf, 'model1.pkl')


