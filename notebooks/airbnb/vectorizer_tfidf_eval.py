#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

filename = '../../data/airbnbdata/201702_listings_preproc.csv'
df = pd.read_csv(filename,
                 encoding='utf-8',
                 sep=',',
                 header=0)

# Isolate rows with known Zip
df = df[~pd.isnull(df.zipcode_new)]
# print('# known zip: {}'.format(len(df)))

# select column subset
df = df[['street', 'zipcode_new']]
df.head()

# keep only the street name portion 
df['street'] = df['street'].apply(lambda x: x.split(',')[0])
df.head()

# init tf-idf
tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')

# train/test split
X = df.copy()
y = X.pop('zipcode_new')

cv_scores, cv_scores2, lost = [], [], []
total = 0
nbr_folds = 10
cv = KFold(n_splits=nbr_folds, random_state=42, shuffle=False)

for train_index, test_index in cv.split(X):
    
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = (X.iloc[train_index], 
                                        X.iloc[test_index], 
                                        y.iloc[train_index], 
                                        y.iloc[test_index])
    
    # reshape y_train
    y_train = y_train.values.reshape((len(y_train), 1))

    # use X_train to compile Document-Term Matrix M
    corpus = X_train.street.values.tolist()
    
    # Transforms the data into a bag of words
    train_vocab = tf.fit(corpus)
    M = tf.transform(corpus)
    
    ## use X_test to generate Matrix W
    q = X_test.street.values.tolist()
    q[:5]
    W = tf.transform(q).todense()
    W[:5, :]

    ## matrix mult. to get scores
    R = M @ W.T
    R[:5, :]

    ## argmax(): get the index of the largest value in each column of R
    # Each column in R represents the multiplication of the corpus tfidf matrix with the a given vector reprsentation of a query address.
    ix = np.argmax(R, axis=0) # axis=0 > run through each column
    # flatten list of lists
    ix = [item for sublist in ix.tolist() for item in sublist]
    ix[:10]

    ## max(): get the largest value in each column of R
    scores = pd.DataFrame(R).max().values.tolist()
    scores[:5]

    ## lookup most similar Zipcode
    # * The index of the largest value in R corresponds to the row in X which is most similar to the query address. 
    # * Thus we can get the most similar Zipcode from y_train.
    # * Note that we have a prediction for each of the unknown data points.
    y_pred = pd.DataFrame(y_train).iloc[ix, 0].values
    y_pred[:5]

    X_test['zipcode'] = y_test
    X_test['zipcode_pred'] = y_pred
    X_test['scores'] = scores
    X_test.head()

    cv_scores.append(accuracy_score(y_true=y_test, y_pred=y_pred))
    
    threshold = 20
    Xi_test = X_test[X_test.scores>threshold].copy()
    y_test = Xi_test.zipcode.values.tolist()
    y_pred = Xi_test.zipcode_pred.values.tolist()
    cv_scores2.append(accuracy_score(y_true=y_test, y_pred=y_pred))
    lost.append(len(X_test)-len(Xi_test))
    total = len(X_test)
    
            
print('Average Accuracy for {}-fold Cross Validation: {}'
      .format(nbr_folds, 
              np.round(np.mean(cv_scores), 3)))

print('Average Accuracy w/threshold {} for {}-fold Cross Validation: {} with an Avg. of {} lost data points out of {}'
      .format(threshold, 
              nbr_folds, 
              np.round(np.mean(cv_scores2), 3), 
              np.round(np.mean(lost), 3), 
              total))
