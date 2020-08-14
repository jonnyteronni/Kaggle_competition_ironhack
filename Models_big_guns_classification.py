'''
Models preparation for competition
'''

#from cleaning.py import import_data
import pandas as pd

# Get dataset
from sklearn.datasets import load_breast_cancer

df=pd.read_csv('Fraud_DB.csv')
# df = pd.DataFrame(import_data())

# Split of data set

#fit
target = df['isFraud']
features = df.drop('isFraud',axis=1)
features=features.select_dtypes(exclude=['object'])


# features, target = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2,random_state=0)

'''
All models for competition
'''




# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



RF_model = RandomForestClassifier(n_jobs=3, random_state=0)
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

# compare predictions to actual answers
print('Confusion matrix')
print(confusion_matrix(y_pred,y_test))
print('-------------------------------------------------------')

# accuracy_score
# fitted X_test data vs. y_test data (actual answer)
print('Accuracy score')
print(accuracy_score(y_pred,y_test))
print('-------------------------------------------------------')

# classification report
print('Classification report')
print(classification_report(y_pred,y_test))








#-----------------------------------------------------------------------------#
# ADA boosting

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# initialize model
DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)

ADA = AdaBoostClassifier(base_estimator = DTC)

# fit model
ADA.fit(X_train, y_train)

#predictions: test data
y_pred = ADA.predict(X_test)

print('\n\nADA boosting report')
#Scores
print('Train score')
print(ADA.score(X_train, y_train))
print('Test score')
print(ADA.score(X_test, y_test))
print('-------------------------------------------------------')

# compare predictions to actual answers
print('Confusion matrix')
print(confusion_matrix(y_pred,y_test))
print('-------------------------------------------------------')

# accuracy_score
# fitted X_test data vs. y_test data (actual answer)
print('Accuracy score')
print(accuracy_score(y_pred,y_test))
print('-------------------------------------------------------')

# classification report
print('Classification report')
print(classification_report(y_pred,y_test))









#-----------------------------------------------------------------------------#
# Gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

# initialize model
GB_model = GradientBoostingClassifier(n_estimators=20, learning_rate=1, max_features=2, max_depth=2, random_state=0)

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

# compare predictions to actual answers
print('Confusion matrix')
print(confusion_matrix(y_pred,y_test))
print('-------------------------------------------------------')

# accuracy_score
# fitted X_test data vs. y_test data (actual answer)
print('Accuracy score')
print(accuracy_score(y_pred,y_test))
print('-------------------------------------------------------')

# classification report
print('Classification report')
print(classification_report(y_pred,y_test))




#-----------------------------------------------------------------------------#
# XGBoost
from xgboost import XGBClassifier

# initialize model
xgb_clf = XGBClassifier()
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

# compare predictions to actual answers
print('Confusion matrix')
print(confusion_matrix(y_pred,y_test))
print('-------------------------------------------------------')

# accuracy_score
# fitted X_test data vs. y_test data (actual answer)
print('Accuracy score')
print(accuracy_score(y_pred,y_test))
print('-------------------------------------------------------')

# classification report
print('Classification report')
print(classification_report(y_pred,y_test))
