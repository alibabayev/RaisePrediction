import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


df = pd.read_csv('/home/stajyer/Desktop/RaisePrediction/dataAddedLMHFinal.csv')
df.head()
df.isnull().sum()

df['TotalQ'] = df['LevelOfHike']
df['TotalQ'].loc[df.TotalQ == 'L'] = 0.0
df['TotalQ'].loc[df.TotalQ == 'M'] = 1.0
df['TotalQ'].loc[df.TotalQ == 'H'] = 2.0

X = df.ix[:,[0,1,2,3,4,6,7,9,10,12,13,14,15,16,18,25,30,33,34,35]]
y = np.array(df['TotalQ'])


X = pd.get_dummies(X)


X.shape


df['Totall'] = df['LevelOfHike']
df['Totall'].loc[df.Totall == 'L'] = 0.0
df['Totall'].loc[df.Totall == 'M'] = 1.0
df['Totall'].loc[df.Totall == 'H'] = 2.0

#continuous_subset = df.ix[:,[0,1,2,3,4,6,7,10,12,13,14,15,20,22,25,29,30,31,32,33,34,35]]
#one_hot_encoded_training_predictors = pd.get_dummies(continuous_subset)
#continuous_subset = one_hot_encoded_training_predictors.ix[:,[0,2,3,4,6,7,10,12,13,14,15,16,18,20,22,25,29,30,31,32,33,34,35]]
#continuous_subset = pd.get_dummies(continuous_subset)
X = df.ix[:,[0,1,2,3,4,6,7,10,12,13,14,15,16,18,25,28,33,34,35]]
y = np.array(df['Totall'])


X = pd.get_dummies(X)


X.shape

names = ['DecisionTreeRegressor', 'LinearRegression', 'Ridge', 'Lasso','XGBRegressor']

clf_list = [DecisionTreeRegressor(),
            LinearRegression(),
            Ridge(),
            Lasso(),XGBRegressor()]

for name, clf in zip(names, clf_list):
    print(name, end=': ')
    print(cross_val_score(clf, X, y, cv=5).mean())

tree = DecisionTreeRegressor()
tree.fit(X, y)

importances = tree.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print("%d. Feature %s (%f)" % (f + 1, X.columns.values[indices[f]], importances[indices[f]]))



X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.35, random_state=0)

 
sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
#print('Misclassified samples: %d' % (y_test != y_pred).sum()


print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


svm = SVC(kernel='linear', C=2.0, random_state=0)
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, y_test)))



print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

my_model = XGBRegressor()
my_model.fit(X_train_std, y_train, verbose=False)
y_pred = my_model.predict(X_test_std)
print('Misclassified samples for XGBoost: %d' % (y_test != y_pred).sum())
print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, y_test)))

print("XGBoost was no better than SVC")
