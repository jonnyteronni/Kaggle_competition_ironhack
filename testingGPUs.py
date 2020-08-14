# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:07:11 2020

@author: BlackTemplario
"""

# Regression Example With Boston Dataset: Baseline
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# load dataset
df = read_csv("Fraud_DB.csv")

# dataset = dataframe.values
# split into input (X) and output (Y) variables
target = df['isFraud']
features = df.drop('isFraud',axis=1)
features=features.select_dtypes(exclude=['object'])
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model


estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=2)
results = cross_val_score(estimator, features, target, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))