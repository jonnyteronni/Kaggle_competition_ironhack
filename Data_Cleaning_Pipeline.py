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

# CREATE DUMMIES

data_w_dummies=pd.get_dummies(data)

data_w_dummies.drop(columns="State_holiday",inplace=True)

data_w_dummies.columns=['Unnamed: 0', 'Store_ID', 'Day_of_week', 'Date', 'Nb_customers_on_day',
       'Open', 'Promotion', 'School_holiday', 'State_holiday_0',
       'State_holiday_a', 'State_holiday_b', 'State_holiday_c', 'Sales']

heatmap_f(data_w_dummies)


#Create SPLIT

X = data_w_dummies.drop(columns=['TARGET COLLUMN'],axis=1)
y = data_w_dummies['TARGET COLLUMN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)





# Check Numerical scale and standardize the numerical values
data.boxplot()

sc=StandardScaler()

X_train=pd.DataFrame(sc.fit_transform(X_train))

X_test=pd.DataFrame(sc.transform(X_test))

