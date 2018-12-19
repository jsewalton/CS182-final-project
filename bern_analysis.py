# Sentiment Analysis of Tweets Using Multinomial Naive Bayes
# https://towardsdatascience.com/sentiment-analysis-of-tweets-using-multinomial-naive-bayes-1009ed24276b


import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


tweets_data_path = '/Users/jessicaedwards/CS182_final_project/train/ExtractedTweets.csv'

tweets = pd.read_csv(tweets_data_path, header=0)

df = tweets.copy()[['party', 'text']]

n_tweet = 42251

# Divide into number of classes
df_dem = df.copy()[df.party == 'Democrat'][:n_tweet]    
df_rep = df.copy()[df.party == 'Republican'][:n_tweet]

df = pd.concat([df_dem, df_rep], ignore_index=True).reset_index(drop=True)

def preprocess_text(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text)
    text = re.sub('@[^\s]+','USER', text)
    text = re.sub(' +',' ', text)
    return text.strip()

data = [preprocess_text(t) for t in df]

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', lowercase=True)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', BernoulliNB())]) #BernoulliNB

tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}

X_train, X_test, y_train, y_test = train_test_split(
	df.text, df.party, test_size=0.33, random_state=0)


score = 'f1_macro'
print("# Tuning hyper-parameters for %s" % score)
print()
np.errstate(divide='ignore')
clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring=score)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
for mean, std, params in zip(clf.cv_results_['mean_test_score'], 
                             clf.cv_results_['std_test_score'], 
                             clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
print(classification_report(y_test, clf.predict(X_test), digits=4))
print()




