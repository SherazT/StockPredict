import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]


df['Percent_Volatility'] = ((df['Adj. High'] - df['Adj. Low'])/ df['Adj. Low'])* 100.0

df['Percent_Change'] = ((df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'])* 100.0

df = df[['Adj. Close', 'Percent_Volatility', 'Percent_Change', 'Adj. Volume']]

forecast_column = 'Adj. Close'

df.fillna(-99999, inplace=True) #replace NaN values with a really small number to make it an outlier instead

forecast_offset = int(math.ceil(0.01*len(df))) #predict 10% of dataframe

df['prediction'] = df[forecast_column].shift(-forecast_offset) #initilize prediction column with data from 10% days ago

print(df.head())