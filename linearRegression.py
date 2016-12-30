import pandas as pd
import quandl, time
import math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import pdb
import matplotlib.pyplot as plt 
from matplotlib import style

style.use('ggplot')

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

#pdb.set_trace() #byebug sub for python


df.dropna(inplace=True) #X no longer contains rows that X_lately has 

#predictions/labels
y = np.array(df['prediction_for_10_days_after'])

#setup training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#setup classifer 
classifer = LinearRegression()
classifer.fit(X_train, y_train) #run gradient decent and fit parameters 
accuracy = classifer.score(X_test, y_test)

forecast_set = classifer.predict(X_lately) #based on these inputs X, find y. currently no known y values

#print(forecast_set, accuracy, forecast_offset)

df['Forecast'] = np.nan #empty column

last_date = df.iloc[-1].name #last date in dataframe (one of the dates that was predicted in line 55)
last_unix = time.mktime(last_date.to_pydatetime().timetuple())	#unix timestamp of the last date 
one_day = 86400	#no of seconds in a day
next_unix = last_unix + one_day	

pdb.set_trace()

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
