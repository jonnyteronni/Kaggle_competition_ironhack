'''
Models preparation for competition
'''

#from cleaning.py import import_data
import pandas as pd

# Get dataset for testing
from sklearn.datasets import load_breast_cancer

#df = pd.DataFrame(import_data())

# Split of data set

#fit
# target = df['target']
# features = df.drop('target',axis=1)

from sklearn.model_selection import train_test_split

features, target = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2,random_state=0)

'''
All models for competition
'''

'''
Classification models
'''

# Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

logistic_model = LogisticRegression(max_iter=2000)
logistic_model.fit(X_train, y_train)

#predictions: test data
y_pred = logistic_model.predict(X_test)

print('\nLogistic Regression report')
#Scores
print('Train score')
print(logistic_model.score(X_train, y_train))
print('Test score')
print(logistic_model.score(X_test, y_test))
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
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

# initialize model
tree_model = DecisionTreeClassifier()

# fit model
tree_model.fit(X_train, y_train)

#predictions: test data
y_pred = tree_model.predict(X_test)

print('\n\nDecision Tree report')
#Scores
print('Train score')
print(tree_model.score(X_train, y_train))
print('Test score')
print(tree_model.score(X_test, y_test))
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
# Support Vector Machines (SVM)
from sklearn.svm import LinearSVC

# initialize model
SVM_model = LinearSVC()

# fit model
SVM_model.fit(X_train, y_train)

#predictions: test data
y_pred = SVM_model.predict(X_test)

print('\n\nSVM report')
#Scores
print('Train score')
print(SVM_model.score(X_train, y_train))
print('Test score')
print(SVM_model.score(X_test, y_test))
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
# K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

# initialize model
KNN_model = KNeighborsClassifier(n_neighbors = 3)

# fit model
KNN_model.fit(X_train, y_train)

#predictions: test data
y_pred = KNN_model.predict(X_test)

print('\n\nKNN report')
#Scores
print('Train score')
print(KNN_model.score(X_train, y_train))
print('Test score')
print(KNN_model.score(X_test, y_test))
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
