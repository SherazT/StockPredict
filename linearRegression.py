import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import pdb

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]


df['Percent_Volatility'] = ((df['Adj. High'] - df['Adj. Low'])/ df['Adj. Low'])* 100.0

df['Percent_Change'] = ((df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'])* 100.0

df = df[['Adj. Close', 'Percent_Volatility', 'Percent_Change', 'Adj. Volume']]

forecast_column = 'Adj. Close'

df.fillna(-99999, inplace=True) #replace NaN values with a really small number to make it an outlier instead

forecast_offset = int(math.ceil(0.01*len(df))) #predict 10% of dataframe


df['prediction_for_10_days_after'] = df[forecast_column].shift(-forecast_offset) #initilize prediction column by shifting Adj. Close column 10% up

#features
X = np.array(df.drop(['prediction_for_10_days_after'], axis=1)) #convert to numpy array and drop prediction column 
#normalize features

X = preprocessing.scale(X)
X_lately = X[-forecast_offset:] #hold all of the inputs that don't have a prediction for 10 days after (it's a NaN value)
X = X[:-forecast_offset:] #holds all the inputs where there is a prediction (just the value shifted 10 days up) for 10 days later

pdb.set_trace()


df.dropna(inplace=True)
#predictions/labels
y = np.array(df['prediction_for_10_days_after'])

#setup training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#setup classifer 
classifer = LinearRegression()
classifer.fit(X_train, y_train) #run gradient decent and fit parameters 
accuracy = classifer.score(X_test, y_test)

