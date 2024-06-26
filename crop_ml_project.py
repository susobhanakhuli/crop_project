# -*- coding: utf-8 -*-
"""Crop-ml-project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HGbXOFFQXqTn5oDkGmqRymFCCt0df8Bo
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stat
import seaborn as sns

import warnings
import pickle
warnings.filterwarnings("ignore")

df = pd.read_csv('Crop_recommendation.csv')
df1 = df.copy()
df

num =  []
cat = []

for i in df.columns:
  if df[i].dtype == 'O':
    cat.append(i)
  else:
    num.append(i)

num

cat

valNull = []

for i in df.columns:
  if df[i].isnull().sum()!=0:
    valNull.append(i)

print(valNull)

"""No Null values

"""

for i in num:
  df1[i].hist()
  plt.show()
  print("\n")

for i in num:
    sns.distplot(df1[i])
    plt.show()

for i in num:
    df1.boxplot(i)
    plt.show()
    print("\n\n")

df1.isnull().sum()

for i in num:
  IQR = df1[i].quantile(0.75) - df1[i].quantile(0.25)

  u_p = df1[i].quantile(0.75) + 1.5*IQR
  l_p = df1[i].quantile(0.25) - 1.5*IQR

  df1.loc[df1[i]>=u_p,i] = u_p
  df1.loc[df1[i]<=l_p,i] = l_p

for i in num:
    df1.boxplot(i)
    plt.show()
    print("\n\n")

box = []
for i in num:
  if (df1[i] <= 0).sum() == 0:
    box.append(i)

box

for i in box:
  data,params=stat.boxcox(df1[i])
  df1[i] = data
  sns.distplot(data)
  plt.show()

df1['label'].unique()

df1['label'].nunique()

from sklearn import preprocessing
# label_encoder object knows
# how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df1['label']= label_encoder.fit_transform(df1['label'])

df1['label'].unique()

"""20- rice
11 - maize
3 - chickpea
9 - kidnebeanks
18 - pigenpeas
13 -  mothbeans
14 - mungbean
2 - bkackgram
10 - lentil
19 - pomegrenate
1 - banana
12 - mango
7 - grapes
21 - watermelon
15 - muskmelon
0 - apple
16 - oraange
17 - papaya
4 - coconut
6 - cotton
8 - jute
5 - coffee
"""

df1.info()

df1.to_csv('file.csv')

X=df1.drop(columns='label',axis=0)
y=df1.label

X

y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5, test_size=0.15)

import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100, max_depth = 5)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# metrics are used to find accuracy or error
from sklearn import metrics

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))

pickle.dump(clf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))