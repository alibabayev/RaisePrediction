import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBRegressor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import preprocessing

df = pd.read_csv('data/dataAddedLMHFinal.csv')
df.head()
df.isnull().sum()


df['Totall'] = df['LevelOfHike']
df['Totall'].loc[df.Totall == 'L'] = 0.0
df['Totall'].loc[df.Totall == 'M'] = 1.0
df['Totall'].loc[df.Totall == 'H'] = 2.0

#continuous_subset = df.ix[:,[0,1,2,3,4,6,7,10,12,13,14,15,20,22,25,29,30,31,32,33,34,35]]

X = df.ix[:,[0,1,2,3,4,6,7,10,12,13,14,15,16,18,20,25,28,31,32,33,34,35]]
y = np.array(df['Totall'])


X = pd.get_dummies(X)

X.shape

names = ['LogisticRegression','RandomForestClassifier','SVC',"RidgeClassifier","AdaBoostClassifier","Perceptron"]

clf_list = [LogisticRegression(),
            RandomForestClassifier(),
            SVC(),RidgeClassifier(),AdaBoostClassifier(),Perceptron()]

print("Cross-Val-Scores")

for name, clf in zip(names, clf_list):
    print(name, end=': ')
    print(cross_val_score(clf, X, y, cv=5).mean())

print()

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.35, random_state=0)


sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#LOGISTIC REGRESSION
logistic = LogisticRegression(multi_class = 'ovr')
logistic.fit(X, y)
y_pred = logistic.predict(X_test_std)

print('Accuracy Logistic: %.2f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print()
print()
print()

print("Predictions", y_pred)
print("Real",y)



#RANDOM FOREST CLASSIFIER
random_for = RandomForestClassifier()
random_for.fit(X, y)
y_pred = random_for.predict(X_test_std)

print('Accuracy Random Forest: %.2f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#SVC
svc = SVC()
svc.fit(X, y)
y_pred = svc.predict(X_test_std)

print('Accuracy SVC: %.2f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#RIDGE
ridge = RidgeClassifier()
ridge.fit(X, y)
y_pred = ridge.predict(X_test_std)

print('Accuracy Ridge: %.2f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#ADABOOST
ada = AdaBoostClassifier()
ada.fit(X, y)
y_pred = ada.predict(X_test_std)

print('Accuracy Ada: %.2f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
