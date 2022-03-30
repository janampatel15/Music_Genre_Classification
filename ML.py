# # ML model implementation
# In this file, we implement different ML classification models to compare it with our Neural Network models (ANN and CNN).
#
# We will run our ML classification models on two csv files. The 30 seconds csv file contains audio features over all of the audio, and 3 seconds csv contains audio feature at every 3 seconds intervals.

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# importing some of our models and pipeline which will help us with performing GridSearch effciently.
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Importing some metrics that'll help us understand and visualize result
from sklearn.metrics import confusion_matrix, classification_report, roc_curve

import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set()

# ## 30 seconds

df = pd.read_csv('./data/features_30_sec.csv')

# converting label columns into category data type
df['label'] = df["label"].astype('category')

# drop all the columns except for mfcc_mean 1-20
df1 = df.drop(df.iloc[:, :19], 1)
df1 = df1.drop(df1.iloc[:, 1::2], 1)
# encode these labels
df1['label'] = df1["label"].cat.codes
df1.shape

df1.head()

# shuffle the dataframe
df1 = df1.sample(frac=1, random_state=0)

# the data scale varies from one mfcc feature to another, so we will normalize it using
# sklearn's statndardscaler

scaler = StandardScaler()
X = df1[df1.columns[~df1.columns.isin(['label'])]].to_numpy()
X = scaler.fit_transform(X)

y = df1['label'].to_numpy()

X.shape, y.shape

# split the dataset, with 30% of dataset used for testing,
# wih random state set to 0, to get same results on each run
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30,
                                                    random_state=0)

# creating pipeline for our gridsearch
pipe = Pipeline(steps=[('classifier', LogisticRegression())])

# defining the models and their hyperparameters that'll be passed into GridSearchCV
param_grid = [{
    'classifier': [LogisticRegression()]
}, {
    'classifier': [GaussianNB()],
    'classifier__var_smoothing': [0.00000001, 0.000000001, 0.0000000001]
}, {
    'classifier': [KNeighborsClassifier()],
    'classifier__metric': ['euclidean', 'manhattan'],
    'classifier__n_neighbors': range(1, 50)
}, {
    'classifier': [RandomForestClassifier()],
    'classifier__n_estimators': range(5, 50, 5),
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_features': ['auto', 'sqrt', 'log2']
}, {
    'classifier': [DecisionTreeClassifier()],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_features': ['auto', 'sqrt', 'log2']
}, {
    'classifier': [SVC()],
    'classifier__kernel': ['linear', 'poly', 'rbf'],
    'classifier__gamma': [0.1, 1, 10, 100]
}]

clf = GridSearchCV(estimator=pipe,
                   param_grid=param_grid,
                   scoring='accuracy',
                   n_jobs=-1,
                   verbose=2,
                   cv=10)

best = clf.fit(X_train, y_train)

# best model and it's hyperparameters
print(f'Our best model and params are {best.best_params_}')

# accuracy score according to the best performing model
print(f'Our Best accurracy score is {round(best.best_score_*100,2)}%')

y_pred = best.predict(X_test)
print(classification_report(y_test, y_pred))

fig = plt.figure(figsize=(8, 6))

cfn_matrix = confusion_matrix(y_test, y_pred)

# for axis tick labels
ticks = df['label'].unique()

sns.heatmap(cfn_matrix, annot=True, xticklabels=ticks, yticklabels=ticks)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)

# ## 3 Seconds

# The steps we took in 30 seconds dataframe (or df dataframe), we repeat those same steps for 3 seconds csv/dataframe

df_short = pd.read_csv('./data/features_3_sec.csv')

# converting label columns into category data type
df_short['label'] = df_short["label"].astype('category')

# drop all the columns except for mfcc_mean 1-20
df_short1 = df_short.drop(df_short.iloc[:, :19], 1)
df_short1 = df_short1.drop(df_short1.iloc[:, 1::2], 1)

# encode these labels
df_short1['label'] = df_short1["label"].cat.codes

# shuffle data
df_short1 = df_short1.sample(frac=1, random_state=0)

# normalizing the data
scaler = StandardScaler()

X_short = df_short1[
    df_short1.columns[~df_short1.columns.isin(['label'])]].to_numpy()
X_short = scaler.fit_transform(X_short)

y_short = df_short1['label'].to_numpy()

X_short.shape, y_short.shape

# splitting dataset
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_short,
                                                        y_short,
                                                        test_size=0.30,
                                                        random_state=0)

best = clf.fit(X_train1, y_train1)

# best performing models and its hyperparameters
print(f'Our best model and params are {best.best_params_}')

# accuracy score according to the best performing model
print(f'Our Best accurracy score is {round(best.best_score_*100,2)}%')

y_pred1 = best.predict(X_test1)
print(classification_report(y_test1, y_pred1))

# confusion matrix of 3 seconds dataframe
fig = plt.figure(figsize=(8, 6))
cfn_matrix1 = confusion_matrix(y_test1, y_pred1)

sns.heatmap(cfn_matrix1,
            annot=True,
            xticklabels=ticks,
            yticklabels=ticks,
            fmt="d")
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
