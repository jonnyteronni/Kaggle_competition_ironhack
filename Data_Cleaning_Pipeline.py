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



data = pd.read_csv("train.csv") ####### Insert file path

#Checking for isnull and ratio of nulls VS total nº of rows of the dataset

nulls=pd.DataFrame(data.isnull().sum())
nulls["ratio"]=nulls[0]/data.shape[0]

print(data.info())

#Double check to numeric

cols=['Unnamed: 0', 'Store_ID', 'Day_of_week', 'Nb_customers_on_day',
       'Open', 'Promotion', 'School_holiday', 'Sales']
data[cols]=data[cols].apply(pd.to_numeric)



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


# DAtetime

data["Date"]=data["Date"].apply(pd.to_datetime)

data["Date"]=data["Date"].apply(lambda x: int(x.strftime('%Y%m%d')))

data["Date_year"]=data["Date"].apply(lambda x: int(str(x)[0:4]))
data["Date_month"]=data["Date"].apply(lambda x: int(str(x)[4:6]))
data["Date_day"]=data["Date"].apply(lambda x: int(str(x)[6:8]))


#Drop row's where OPEN = 0

data=data.drop(data[(data["Open"]==0)].index)

data.reset_index(drop=True,inplace=True)


                     
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


#Drop row's where OPEN = 0

#data_w_dummies=data_w_dummies.drop(data[(data["Open"]==0)].index)



#data_w_dummies.reset_index(drop=True,inplace=True)




#Create SPLIT

X = data_w_dummies.drop(columns=['Sales'],axis=1)
y = data_w_dummies['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


# Check Numerical scale and standardize the numerical values
#data_w_dummies.boxplot()

sc=StandardScaler()


X_train_scaled=pd.DataFrame(sc.fit_transform(X_train))
X_train_scaled.columns= X_train.columns

X_train_scaled=pd.concat([X_train_scaled[['Unnamed: 0', 'Store_ID', 'Day_of_week', 'Nb_customers_on_day',
       'Open', 'Promotion', 'School_holiday', 'Date_year', 'Date_month','Date_day']],X_train[['State_holiday_0',
       'State_holiday_a', 'State_holiday_b', 'State_holiday_c']].reset_index(drop=True)],axis=1)


X_test_scaled=pd.DataFrame(sc.transform(X_test))
X_test_scaled.columns=X_test.columns

X_test_scaled=pd.concat([X_test_scaled[['Unnamed: 0', 'Store_ID', 'Day_of_week', 'Nb_customers_on_day',
       'Open', 'Promotion', 'School_holiday', 'Date_year', 'Date_month','Date_day']],X_test[['State_holiday_0',
       'State_holiday_a', 'State_holiday_b', 'State_holiday_c']].reset_index(drop=True)],axis=1)

y_train_scaled=sc.fit_transform(pd.DataFrame(y_train))
y_test_scaled=sc.transform(pd.DataFrame(y_test))



#Linear Regression Model - REGRESSION exercise
model=LinearRegression()
model.fit(X_train_scaled,y_train_scaled)

# Function to score a REGRESSION exercise
def regression_score(model=model, y_test=y_test, X_test=X_test):
    y_pred=model.predict(X_test)
    coefficient_of_dermination = r2_score(y_test,y_pred)
    meanabsoluterror=mean_absolute_error(y_test,y_pred)
    
    print("coefficient_of_dermination (R2), best score is 1.0 ")
    print(coefficient_of_dermination)
    print('-----------------------------------------------')
    print("Mean Absolute Error (MAE) - best score is 0.0")
    print(meanabsoluterror)
    return

regression_score(X_test=X_test_scaled,y_test=y_test_scaled)



data_w_dummies.to_csv("cleaning_w_dummies.csv",index=False)


# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

X_train=X_train_scaled
y_train=y_train_scaled

X_test=X_test_scaled
y_test=y_test_scaled

RF_model = RandomForestRegressor(n_jobs=6, random_state=0)
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
