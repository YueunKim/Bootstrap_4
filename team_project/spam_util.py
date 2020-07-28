import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import accuracy_score  
from sklearn.model_selection import train_test_split
#%matplotlib inline


def spam_mail_util(app, n, filename):
    df = pd.read_csv(os.path.join(app.root_path, 'data/'+filename), encoding='latin1')

    del df['Unnamed: 2']
    del df['Unnamed: 3']
    del df['Unnamed: 4']
    df['v1'] = df['v1'].replace(['ham','spam'],[0,1])

    df = df.drop_duplicates('v2', keep='first')

    X_data = df['v2']
    y_data = df['v1']

    X_train, X_test, y_train, y_test = \
        train_test_split(X_data, y_data, test_size=.2, random_state=2020)

    dtmvector = CountVectorizer()
    X_train_dtm = dtmvector.fit_transform(X_train)
    
    model = MultinomialNB()
    model.fit(X_train_dtm, y_train)
    X_test_dtm = dtmvector.transform(X_test)
    predicted = model.predict(X_test_dtm)

    acc = accuracy_score(y_test, predicted)