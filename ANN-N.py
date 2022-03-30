#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network
# In this file, we implement different ANN classification models to compare it with our Machine Learning and CNN models.
#
# We will run our ANN classification models on two csv files. The 30 seconds csv file containing audio features over for the whole audio, and 3 seconds csv contains audio feature at every 3 seconds intervals.

# important modules
import numpy as np

np.random.seed(0)
import os
import warnings

import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from nis_new_main_class import *
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings('ignore')
import seaborn as sns

sns.set()

# ## 30 Seconds

df = pd.read_csv('./data/features_30_sec.csv')
df.head()

df['label'] = df["label"].astype('category')
df['label'] = df["label"].cat.codes

df1 = df.drop(df.iloc[:, :19], 1)
df1 = df1.drop(df1.iloc[:, 1::2], 1)

df1.head()

# goood to see if there is any major correlation, nulls or any other key factors that we might miss
df_profile = df1[df1.columns[~df1.columns.isin(['label'])]]

# When ran the below commented line, the file is big (50 MB) so careful when opening/running it
# profile = ProfileReport(df_profiel)

# line below does about the same job, except it leaves our the scatter matrix between each feature
# allowing us to have smaller report html file.
profile = ProfileReport(df_profile, interactions={'continuous': False})

profile.to_file("30sec_report.html")

# #### Reason for Duplicates
# From the report generated above or [here](30sec_report.html), we see that we don't have nulls, but we do have some duplicates, it could be because a lot of mfcc features differ mostly in thousandth place or after, so alot of decimal places do end up matching which gives us duplicates. Also, there are only 7 signals with each having 1 duplicate, giving us 7 duplicates out of 1000, so not too bad.

# We will shuffle dataframe even though we have train_test_split, just to be on the safe side.
df1 = df1.sample(frac=1, random_state=0)

# First seperating features and labels
# X = Features
# y = labels
X = df1[df1.columns[~df1.columns.isin(['label'])]].to_numpy()
y = df1['label'].to_numpy()

# split the dataset wiht 30% set aside for the testing purpose, and random_state set to 0,
# for same reproducibility
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30,
                                                    random_state=0)

X_train.shape, y_train.shape

# one hot encoding our labels
y_true_train = one_hot(y_train, 10)
y_true_test = one_hot(y_test, 10)

# creating layers to pass it into our models
outs = 20
outs1 = 20
lay_train = [
    InputLayer(X_train),
    FullyConnectedLayer(len(X_train[0]), outs),
    TanhLayer(),
    FullyConnectedLayer(outs, outs1),
    TanhLayer(),
    FullyConnectedLayer(outs1, 10),
    SigmoidLayer(),
    LogLoss()
]

layers_val = [
    InputLayer(X_test),
    FullyConnectedLayer(len(X_test[0]), outs),
    TanhLayer(),
    FullyConnectedLayer(outs, outs1),
    TanhLayer(),
    FullyConnectedLayer(outs1, 10),
    SigmoidLayer(),
    LogLoss()
]

# running those given layers with our model function with specific given eta and epochs
# the eta and epochs shown below aren't final and are just shown as a demo to check if code works correctly

ll_train, y_pred_train, ll_val, y_pred_val, acc_train, acc_val, epochs, eta = model(
    X_train,
    y_train,
    X_test,
    y_test,
    lay_train,
    layers_val,
    eta=0.001,
    epochs=500)

ll_train1, y_pred_train1, ll_val1, y_pred_val1, acc_train1, acc_val1, epochs1, eta1 = model(
    X_train,
    y_train,
    X_test,
    y_test,
    lay_train,
    layers_val,
    eta=0.0001,
    epochs=500)

# the graphs and accuracy you see below aren't the final ones we chose,
# these are just demo runs to check if our code works fine

print('When we train and test our model with 30 seconds dataframe')
print()
print(f'When epochs = {epochs} and eta = {eta}:')
print(
    f'Training Accuracy: {round(acc_train[-1],2)}%, Testing Accuracy: {round(acc_val[-1],2)}%'
)
print()
print(f'When epochs = {epochs1} and eta = {eta1}:')
print(
    f'Training Accuracy: {round(acc_train1[-1],2)}%, Testing Accuracy: {round(acc_val1[-1],2)}%'
)

alls = [
    ll_train, ll_val, acc_train, acc_val, epochs, eta, ll_train1, ll_val1,
    acc_train1, acc_val1, epochs1, eta1
]

fig = plt.figure()
fig.set_size_inches(10, 10)

create_graphs(alls, 3, 2)

# ## 3 seconds
# We will repeat same steps as we did for 30 seconds csv (dataframe)

df_short = pd.read_csv('./data/features_3_sec.csv')
df_short

df_short['label'] = df_short["label"].astype('category')
df_short['label'] = df_short["label"].cat.codes

df1_short = df_short.drop(df_short.iloc[:, :19], 1)
df1_short = df1_short.drop(df1.iloc[:, 1::2], 1)

# goood to see if there is any major correlation, nulls or any other key factors that we might miss
df_profile1 = df1_short[df1_short.columns[~df1_short.columns.isin(['label'])]]

# When ran the below commented line, the file is big (50 MB) so careful when opening/running it
# profile = ProfileReport(df_profiel)

# line below does about the same job, except it leaves our the scatter matrix between each feature
# allowing us to have smaller report html file.
profile1 = ProfileReport(df_profile1, interactions={'continuous': False})

profile1.to_file("3sec_report.html")

# From the report generated above or [here](3sec_report.html), we see that we don't have nulls, but we do have some duplicates, it could be the same reason as we had for 30 seconds dataframe having 14 duplicates. [30seconds_reason](#Reason-for-Duplicates)

df1_short = df1_short.sample(frac=1, random_state=0)

X_short = df1_short[
    df1_short.columns[~df1_short.columns.isin(['label'])]].to_numpy()
y_short = df1_short['label'].to_numpy()

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_short,
                                                        y_short,
                                                        test_size=0.30,
                                                        random_state=0)

y_true_train1 = one_hot(y_train1, 10)
y_true_test1 = one_hot(y_test1, 10)

outs = 20
outs1 = 20

lay_train = [
    InputLayer(X_train1),
    FullyConnectedLayer(len(X_train1[0]), outs),
    TanhLayer(),
    FullyConnectedLayer(outs, outs1),
    TanhLayer(),
    FullyConnectedLayer(outs1, 10),
    SigmoidLayer(),
    LogLoss()
]

layers_val = [
    InputLayer(X_test1),
    FullyConnectedLayer(len(X_test1[0]), outs),
    TanhLayer(),
    FullyConnectedLayer(outs, outs1),
    TanhLayer(),
    FullyConnectedLayer(outs1, 10),
    SigmoidLayer(),
    LogLoss()
]

# running those given layers with our model function with specific given eta and epochs
# the eta and epochs shown below aren't final and are just shown as a demo to check if code works correctly

ll_train2, y_pred_train2, ll_val2, y_pred_val2, acc_train2, acc_val2, epochs2, eta2 = model(
    X_train1,
    y_train1,
    X_test1,
    y_test1,
    lay_train,
    layers_val,
    eta=0.001,
    epochs=500)

ll_train3, y_pred_train3, ll_val3, y_pred_val3, acc_train3, acc_val3, epochs3, eta3 = model(
    X_train1,
    y_train1,
    X_test1,
    y_test1,
    lay_train,
    layers_val,
    eta=0.0001,
    epochs=500)

# the graphs and accuracy you see below aren't the final ones we chose,
# these are just demo runs to check if our code works fine

print('When we train and test our model with 3 seconds dataframe')
print()
print(f'When epochs = {epochs2} and eta = {eta2}:')
print(
    f'Training Accuracy: {round(acc_train2[-1],2)}%, Testing Accuracy: {round(acc_val2[-1],2)}%'
)
print()
print(f'When epochs = {epochs3} and eta = {eta3}:')
print(
    f'Training Accuracy: {round(acc_train3[-1],2)}%, Testing Accuracy: {round(acc_val3[-1],2)}%'
)

#
alls = [
    ll_train2, ll_val2, acc_train2, acc_val2, epochs2, eta2, ll_train3,
    ll_val3, acc_train3, acc_val3, epochs3, eta3
]

fig = plt.figure()
fig.set_size_inches(10, 10)

create_graphs(alls, 3, 2)
