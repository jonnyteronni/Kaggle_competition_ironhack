############ DATA CLEANING  ############
# Imports
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import r2_score, mean_absolute_error



#Data exploration


data = pd.read_csv("DATA HERE!!!!.csv") ####### Insert file path

print(data.info())

print(data.columns)


#Checking for isnull and ratio of nulls VS total nº of rows of the dataset

nulls=pd.DataFrame(data.isnull().sum())
nulls["ratio"]=nulls[0]/data.shape[0]


# Changing all str or object collumn type to UPPERCASE

data=data.apply(lambda x: x.astype(str).str.upper())


# Example if we want to replace str in collumns

dic = {"[U'GB'; U'UK']": "UK", "UNITED KINGDOM" : "UK", "GB": "UK", "CYPRUS": "CY"}

data["COLUMN NAME HERE]=data["COLUMN NAME HERE"].apply(lambda x: x.upper()).replace(dic)
     
#Example of function to create "clean" categories

def join_cat(x):
    if "MICROSOFT" in str(x.upper()):
        ans="Microsoft"
    elif "APACHE" in str(x.upper()):
        ans="Apache"
    elif "NGINX" in str(x.upper()):
        ans="nginx"
    else:
        ans="Other"
    return ans


data["COLUMN NAME HERE"]=data["COLUMN NAME HERE"].astype("str").apply(join_cat)


# CREATE DUMMIES

data_w_dummies=pd.get_dummies(data)



# Check Numerical scale and standardize the numerical values
data.boxplot()

sc=StandardScaler()

X_train=pd.DataFrame(sc.fit_transform(X_train))

X_test=pd.DataFrame(sc.transform(X_test))


#Function that prints the Collinearity

def heatmap_f(dataframe):

    corr = dataframe.corr()

    sns.set(style="white")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})
    return plt.show()


heatmap_f(data)



# Droping collumns (features) + Heatmap

data.drop(columns="COLUMN NAME HERE",inplace=True)

heatmap_f(data)


# DROP NaN from the dataframe

data.dropna(inplace=True)


#Create SPLIT

X = data_w_dummies.drop(columns=['TARGET COLLUMN'],axis=1)
y = data_w_dummies['TARGET COLLUMN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)



#logistics Regression Model - CLASSIFICATION exercise
model = LogisticRegression(max_iter=2000)
model.fit(X_train,y_train)

#Linear Regression Model - REGRESSION exercise
model=LinearRegression()
model.fit(X_train,y_train)


# Function to score a CLASSIFICATION exercise

def classification_score(X_test=X_test, y_test=y_test):
    y_pred = model.predict(X_test)

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
    return

classification_score()

# Function to score a REGRESSION exercise
def regression_score(model=model, y_test=y_test):
    y_pred=model.predict(X_test)
    coefficient_of_dermination = r2_score(y_test,y_pred)
    meanabsoluterror=mean_absolute_error(y_test,y_pred)
    
    print("coefficient_of_dermination (R2), best score is 1.0 ")
    print(coefficient_of_dermination)
    print('-----------------------------------------------')
    print("Mean Absolute Error (MAE) - best score is 0.0")
    print(meanabsoluterror)
    return
regression_score()


#incomplete below
"""from sklearn.model_selection import ShuffleSplit
n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(clf, X, y, cv=cv)
array([0.977..., 0.977..., 1.  ..., 0.955..., 1.        ])""""""
