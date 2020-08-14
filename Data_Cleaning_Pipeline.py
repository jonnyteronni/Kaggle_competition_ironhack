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

from datetime import datetime

from time import time

start=time()


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


def clean_data(csv):
    data = pd.read_csv(csv) ####### Insert file path
    
    #Checking for isnull and ratio of nulls VS total nº of rows of the dataset
    
    nulls=pd.DataFrame(data.isnull().sum())
    nulls["ratio"]=nulls[0]/data.shape[0]
    
    #print(data.info())
    
    #Double check to numeric
    
    cols=['Unnamed: 0', 'Store_ID', 'Day_of_week', 'Nb_customers_on_day',
           'Open', 'Promotion', 'School_holiday', 'Sales']
    data[cols]=data[cols].apply(pd.to_numeric)
    
    
    
   
    
    # DAtetime
    
    data["Date"]=data["Date"].apply(pd.to_datetime)
    
    data["Date"]=data["Date"].apply(lambda x: int(x.strftime('%Y%m%d')))
    
    data["Date_year"]=data["Date"].apply(lambda x: int(str(x)[0:4]))
    data["Date_month"]=data["Date"].apply(lambda x: int(str(x)[4:6]))
    data["Date_day"]=data["Date"].apply(lambda x: int(str(x)[6:8]))
    
                         
    # CREATE DUMMIES
    
    data_w_dummies=pd.get_dummies(data)
    
    #data_w_dummies.drop(columns="State_holiday",inplace=True)
    
    data_w_dummies=data_w_dummies[['Unnamed: 0', 'Store_ID', 'Day_of_week', 'Date', 'Nb_customers_on_day',
           'Open', 'Promotion', 'School_holiday', 'Date_year',
           'Date_month', 'Date_day', 'State_holiday_0', 'State_holiday_a',
           'State_holiday_b', 'State_holiday_c', 'Sales']]
    
    heatmap_f(data_w_dummies)
    
    #1st Drop
    data_w_dummies.drop(columns="Date",inplace=True)
    
    heatmap_f(data_w_dummies)
    
    
    data_w_dummies.drop(columns="State_holiday_a",inplace=True)
    
    heatmap_f(data_w_dummies)
    
    
    #Drop row's where OPEN = 0
    #THIS DROP DID NOT WORK
    #data_w_dummies=data_w_dummies.drop(data[(data["Open"]==0)].index)
    #data_w_dummies.reset_index(drop=True,inplace=True)
    
    #Create SPLIT
    
    X = data_w_dummies.drop(columns=['Sales'],axis=1)
    y = data_w_dummies['Sales']
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    
    
    # Check Numerical scale and standardize the numerical values
    #data_w_dummies.boxplot()
    
    sc=StandardScaler()
    
    #To use if we do a "Normal" Split
    X_train_scaled=pd.DataFrame(sc.fit_transform(X_train))
    X_train_scaled.columns= X_train.columns
    
    X_train_scaled=pd.concat([X_train_scaled[['Unnamed: 0', 'Store_ID', 'Day_of_week', 'Nb_customers_on_day',
            'Open', 'Promotion', 'School_holiday', 'Date_year', 'Date_month','Date_day']],X_train[['State_holiday_0', 'State_holiday_b', 'State_holiday_c']].reset_index(drop=True)],axis=1)
    
    
    X_test_scaled=pd.DataFrame(sc.transform(X_test))
    X_test_scaled.columns=X_test.columns
    
    X_test_scaled=pd.concat([X_test_scaled[['Unnamed: 0', 'Store_ID', 'Day_of_week', 'Nb_customers_on_day',
            'Open', 'Promotion', 'School_holiday', 'Date_year', 'Date_month','Date_day']],X_test[[
            'State_holiday_0', 'State_holiday_b', 'State_holiday_c']].reset_index(drop=True)],axis=1)
    
    y_train_scaled=sc.fit_transform(pd.DataFrame(y_train))
    y_test_scaled=sc.transform(pd.DataFrame(y_test))
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled



#Linear Regression Model - REGRESSION exercise
model=LinearRegression()
model.fit(X_train_scaled,y_train_scaled)

# Function to score a REGRESSION exercise
def regression_score(model=model, y_test=y_test, X_test=X_test,X_train=X_train,y_train=y_train):
    y_pred=model.predict(X_test)
    y_pred_train=model.predict(X_train)
    coefficient_of_dermination = r2_score(y_test,y_pred)
    meanabsoluterror=mean_absolute_error(y_test,y_pred)
    
    coefficient_of_dermination_train = r2_score(y_train,y_pred_train)
    
    print("coefficient_of_dermination TEST (R2), best score is 1.0 ")
    print(coefficient_of_dermination)
    print('-----------------------------------------------')  
    print("coefficient_of_dermination TRAIN (R2), best score is 1.0 ")
    print(coefficient_of_dermination_train)
    print('-----------------------------------------------')
    print("Mean Absolute Error (MAE) - best score is 0.0")
    print(meanabsoluterror)
    return

regression_score(X_test=X_test_scaled,y_test=y_test_scaled,X_train=X_train_scaled,y_train=y_train_scaled)

#SAVING TO CSV CLEAN FILE

data_w_dummies.to_csv("cleaning_w_dummies.csv",index=False)


from sklearn.model_selection import ShuffleSplit



# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


X_train=X_train_scaled
y_train=y_train_scaled

X_test=X_test_scaled
y_test=y_test_scaled

# RF_model = RandomForestRegressor(n_jobs=6, random_state=0)
# RF_model.fit(X_train, y_train)


# lets see the code to perform hyperparameter tunings
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# we are going to apply grid search and randomized search, but I have to define the possible values of the hyperparameters to test

# Number of trees in random forest
n_estimators = [10,100,1000,2000]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in each tree
max_depth = [5,10]


# Create the  grid -> this is a dictionary where you will state each hyperparameters you want to tune, and then feed it to the searching algorithm
# the keys in this dictionary have to match the names of the hyperparameters in the documentation of the model
grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth
               }

# Instantiate the grid search model

# estimator -> what model to optimize 
RF_model = RandomForestRegressor(n_jobs=7)
# param_grid -> state the dictionary of parameters to optimize

# cv = 3 -> number of cross validation folds

grid_search = GridSearchCV(estimator = RF_model, param_grid = grid, cv = 3)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

#
print(grid_search.best_params_)


#RANDOM 
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

grid_search = RandomizedSearchCV(estimator = RF_model, param_distributions = random_grid, n_iter = 3, cv = 3, verbose=2, random_state=0, n_jobs = 7)

print(grid_search.fit(X_train,y_train))
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.score(X_test, y_test))


# n_samples = data_w_dummies.shape[0]
# cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# results=cross_val_score(RF_model, X, y, cv=cv)

# print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))


#result=crossvalscore(RF_model)

#predictions: test data
y_pred = grid_search.predict(X_test)

print('\nRandom Forest report')
#Scores
print('Train score')
print(grid_search.score(X_train, y_train))
print('Test score')
print(grid_search.score(X_test, y_test))
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

print((time()-start))






#PEDRO's VALIDATION 
clean_data(csv) #Insert path csv
#it returns X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
X_train=X_train_scaled
y_train=y_train_scaled

X_test=X_test_scaled
y_test=y_test_scaled



