'''
Models preparation for competition
'''
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


#from cleaning.py import import_data
import pandas as pd

# Get dataset
# from sklearn.datasets import load_boston



df=pd.read_csv('Fraud_DB.csv')
# df = pd.DataFrame(import_data())

# Split of data set

#fit
target = df['isFraud']
features = df.drop('isFraud',axis=1)
features=features.select_dtypes(exclude=['object'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2,random_state=0)

'''
All models for competition
'''




# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


RF_model = RandomForestRegressor(n_jobs=2, random_state=0)
RF_model.fit(X_train, y_train)

#predictions: test data
y_pred = RF_model.predict(X_test)

print('\nRandom Forest report')
#Scores
print('Train score')
print(RF_model.score(X_train, y_train))
print('Test score')
print(RF_model.score(X_test, y_test))
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








#-----------------------------------------------------------------------------#
# ADA boosting

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# initialize model
DTC = DecisionTreeRegressor(random_state = 11, max_features = "auto", max_depth = None)

ADA = AdaBoostRegressor(base_estimator = DTC)

# fit model
ADA.fit(X_train, y_train)

#predictions: test data
y_pred = ADA.predict(X_test)

print('\n\nDecision Tree report')
#Scores
print('Train score')
print(ADA.score(X_train, y_train))
print('Test score')
print(ADA.score(X_test, y_test))
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










#-----------------------------------------------------------------------------#
# Gradient boosting
from sklearn.ensemble import GradientBoostingRegressor

# initialize model
GB_model = GradientBoostingRegressor(n_estimators=20, learning_rate=1, max_features=2, max_depth=2, random_state=0)

# fit model
GB_model.fit(X_train, y_train)

#predictions: test data
y_pred = GB_model.predict(X_test)

print('\n\nGradient boosting report')
#Scores
print('Train score')
print(GB_model.score(X_train, y_train))
print('Test score')
print(GB_model.score(X_test, y_test))
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




#-----------------------------------------------------------------------------#
# XGBoost
from xgboost import XGBRegressor

# initialize model
xgb_clf = XGBRegressor()
# fit model
xgb_clf.fit(X_train, y_train)

#predictions: test data
y_pred = xgb_clf.predict(X_test)

print('\n\n\nXGBoost')
#Scores
print('Train score')
print(xgb_clf.score(X_train, y_train))
print('Test score')
print(xgb_clf.score(X_test, y_test))
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
