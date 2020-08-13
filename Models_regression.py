'''
Regression models preparation for competition
'''

#from cleaning.py import import_data
import pandas as pd

# Get dataset for testing
from sklearn.datasets import load_boston
# df = pd.DataFrame(import_data())

# Split of data set

#fit
# target = df['target']
# features = df.drop('target',axis=1)

features, target = load_boston(return_X_y=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2,random_state=0)

'''
All models for competition
'''

'''
------------------------------------------------------------------
Regression models
'''

# Linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

#predictions: test data
y_pred = linear_model.predict(X_test)

print('\nLogistic Regression report')
#Scores
print('Train score')
print(linear_model.score(X_train, y_train))
print('Test score')
print(linear_model.score(X_test, y_test))
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
# Ridge regression
from sklearn.linear_model import Ridge

ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

#predictions: test data
y_pred = ridge_model.predict(X_test)

print('\n\n\nRidge Regression report')
#Scores
print('Train score')
print(ridge_model.score(X_train, y_train))
print('Test score')
print(ridge_model.score(X_test, y_test))
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
# Lasso regression
from sklearn.linear_model import Lasso

lasso_model = Lasso(max_iter=2000)
lasso_model.fit(X_train, y_train)

#predictions: test data
y_pred = lasso_model.predict(X_test)

print('\n\n\nLasso Regression report')
#Scores
print('Train score')
print(lasso_model.score(X_train, y_train))
print('Test score')
print(lasso_model.score(X_test, y_test))
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
# Decision Tree

from sklearn.tree import DecisionTreeRegressor

# initialize model
tree_model = DecisionTreeRegressor()

# fit model
tree_model.fit(X_train, y_train)

#predictions: test data
y_pred = tree_model.predict(X_test)

print('\n\n\nDecision Tree report')
#Print the tree
# from sklearn.tree.export import export_text
# r = export_text(tree_model, feature_names=list(boston.feature_names))
# print(r)

#Scores
print('Train score')
print(tree_model.score(X_train, y_train))
print('Test score')
print(tree_model.score(X_test, y_test))
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
# Support Vector Regression (SVR)
from sklearn.svm import LinearSVR

# initialize model
SVR_model = LinearSVR()

# fit model
SVR_model.fit(X_train, y_train)

#predictions: test data
y_pred = SVR_model.predict(X_test)

print('\n\n\nSVM report')

#Scores
print('Train score')
print(SVR_model.score(X_train, y_train))
print('Test score')
print(SVR_model.score(X_test, y_test))
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
# K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsRegressor

# initialize model
KNR_model = KNeighborsRegressor(n_neighbors = 3)

# fit model
KNR_model.fit(X_train, y_train)

#predictions: test data
y_pred = KNR_model.predict(X_test)

print('\n\n\nKNR report')

#Scores
print('Train score')
print(KNR_model.score(X_train, y_train))
print('Test score')
print(KNR_model.score(X_test, y_test))
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
