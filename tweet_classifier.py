# A Naive Bayes Tweet Classifier
# https://www.kaggle.com/degravek/a-naive-bayes-tweet-classifier/notebook

from __future__ import division
from sklearn.model_selection import train_test_split
from stop_words import get_stop_words
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import string
import re

# Read in tweet data
tweets_data_path = '/Users/jessicaedwards/CS182_final_project/train/ExtractedTweets.csv'

tweets = pd.read_csv(tweets_data_path, header=0)

df = tweets.copy()[['party', 'text']]

# Define number of classes and number of tweets per class
n_class = 2
n_tweet = 42251

# Divide into number of classes
if n_class == 2:
    df_dem = df.copy()[df.party == 'Democrat'][:n_tweet]
    df_rep = df.copy()[df.party == 'Republican'][:n_tweet]
    df = pd.concat([df_dem, df_rep], ignore_index=True).reset_index(drop=True)

# Define functions to process Tweet text and remove stop words
def ProTweets(tweet):
    tweet = ''.join(c for c in tweet if c not in string.punctuation)
    tweet = re.sub('((www\S+)|(http\S+))', 'urlsite', tweet)
    tweet = re.sub(r'\d+', 'contnum', tweet)
    tweet = re.sub(' +',' ', tweet)
    tweet = tweet.lower().strip()
    return tweet

def rmStopWords(tweet, stop_words):
    text = tweet.split()
    text = ' '.join(word for word in text if word not in stop_words)
    return text

# Get list of stop words
stop_words = get_stop_words('english')
stop_words = [''.join(c for c in s if c not in string.punctuation) for s in stop_words]
stop_words = [t.encode('utf-8') for t in stop_words]

# Preprocess all tweet data
pro_tweets = []
for tweet in df['text']:
    processed = ProTweets(tweet)
    pro_stopw = rmStopWords(processed, stop_words)
    pro_tweets.append(pro_stopw)

df['text'] = pro_tweets

# Set up training and test sets by choosing random samples from classes
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['party'], test_size=0.33, random_state=0)

df_train = pd.DataFrame()
df_test = pd.DataFrame() 

df_train['text'] = X_train
df_train['party'] = y_train
df_train = df_train.reset_index(drop=True)

df_test['text'] = X_test
df_test['party'] = y_test
df_test = df_test.reset_index(drop=True)


# Start training (input training set df_train)
class TweetNBClassifier(object):

    def __init__(self, df_train):
        self.df_train = df_train
        self.df_dem = df_train.copy()[df_train.party == 'Democrat']
        self.df_rep = df_train.copy()[df_train.party == 'Republican']

    def fit(self):
        Pr_dem = df_dem.shape[0]/self.df_train.shape[0]
        Pr_rep = df_rep.shape[0]/self.df_train.shape[0]
        
        self.Prior = (Pr_dem, Pr_rep)

        self.dem_words = ' '.join(self.df_dem['text'].tolist()).split()
        self.rep_words = ' '.join(self.df_rep['text'].tolist()).split()

        all_words = ' '.join(self.df_train['text'].tolist()).split()

        self.vocab = len(Counter(all_words))

        wc_dem = len(' '.join(self.df_dem['text'].tolist()).split())
        wc_rep = len(' '.join(self.df_rep['text'].tolist()).split())
        
        self.word_count = (wc_dem, wc_rep)
        return self

    def predict(self, df_test):

        class_choice = ['Democrat', 'Republican']

        classification = []
        for tweet in df_test['text']:
            text = tweet.split()

            val_dem = np.array([])
            val_rep = np.array([])

            for word in text:
                tmp_dem = np.log((self.dem_words.count(word)+1)/(self.word_count[0]+self.vocab))
                tmp_rep = np.log((self.rep_words.count(word)+1)/(self.word_count[1]+self.vocab))
                val_dem = np.append(val_dem, tmp_dem)
                val_rep = np.append(val_rep, tmp_rep)

            val_dem = np.log(self.Prior[0]) + np.sum(val_dem)
            val_rep = np.log(self.Prior[1]) + np.sum(val_rep)

            probability = (val_dem, val_rep)

            classification.append(class_choice[np.argmax(probability)])
        return classification


    def score(self, feature, target):

        compare = []
        for i in range(0,len(feature)):
            if feature[i] == target[i]:
                tmp ='correct'
                compare.append(tmp)
            else:
                tmp ='incorrect'
                compare.append(tmp)
        r = Counter(compare)
        accuracy = r['correct']/(r['correct']+r['incorrect'])
        return accuracy

tnb = TweetNBClassifier(df_train)
tnb = tnb.fit()
predict = tnb.predict(df_test)
score = tnb.score(predict,df_test.party.tolist())
print(score)