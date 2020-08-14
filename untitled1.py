# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:27:27 2020

@author: BlackTemplario
"""

# Regression Example With Boston Dataset: Baseline
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

features, target = load_boston(return_X_y=True)
# # load dataset
# dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
# dataset = dataframe.values
# # split into input (X) and output (Y) variables
# X = dataset[:,0:13]
# Y = dataset[:,13]
# define base model



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time

X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2,random_state=0)



def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model

start = time.time()


model = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

model.fit(X_train, y_train)

#predictions: test data
y_pred = model.predict(X_test)

print('\nRandom Forest report')
#Scores
print('Train score')
print(model.score(X_train, y_train))
print('Test score')
print(model.score(X_test, y_test))
print('-------------------------------------------------------')

# MAE
print('Mean absolute error')
print(mean_absolute_error(y_test, y_pred))
print('-------------------------------------------------------')

# MSE
print('Mean squared error')
print(mean_squared_error(y_test, y_pred))
print('-------------------------------------------------------')

# R-squared
print('R-squared')
print(r2_score(y_test, y_pred))

print("Ran in {} seconds".format(time.time() - start))


# kfold = KFold(n_splits=10)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))