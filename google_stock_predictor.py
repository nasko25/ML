import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
#from datetime import datetime

# Got the data from yahoo
data = pd.read_csv('datasets/GOOG.csv')

actual_data = data[-2:].get('Close').tolist()

data.drop(data.tail(2).index, inplace=True)  # drop the last 2 entries to guess them later
data.Date = pd.to_datetime(data.Date)        # convert date to datetime object
data = data.sort_values('Date')              # sort by the date
data.set_index(data.Date, inplace=True)      # Set the index to date
data.drop('Date', 1, inplace=True)	     # Remove the Date column as it was set as an index and is no longer needed as a separate column 

# exog is a variable that the endog(data.Close) depends on
model = ARIMA(data.Close, order=(1,0,3), exog = data.drop(['Volume', 'Close', 'Adj Close'], axis=1))
#model = ARIMA(data.Close, order=(1,0,3), exog = data.drop(['Close', 'Adj Close'], axis=1))

print(data.isnull().any())

model_fit = model.fit()
model_fit.summary()

#print(model_fit.predict(start=1000, end=1001))
# Cannot really predict the future values, because the data is not consistent. Some days are skipped, so the model cannot predict the future based on this dates.
#print(model_fit.predict("2020-03-23", "2020-03-24"))
					        # open   # high   # low    # volume
#print(model_fit.forecast(steps=2,exog=np.array([[1061.32, 1071.32, 1013.54, 4044100], [1103.77, 1135.00, 1090.62, 3341600]]))[0])
#print(model_fit.forecast(steps=2,exog=np.array([1061.32, 1071.32, 1013.54, 4044100]))[0])

print("-------------- NO VOLUME ---------------------")
print("predicted values: {}".format(model_fit.forecast(steps=2,exog=np.array([[1061.32, 1071.32, 1013.54], [1103.77, 1135.00, 1090.62]]))[0]))
print("actual vaules: {}".format(actual_data))

