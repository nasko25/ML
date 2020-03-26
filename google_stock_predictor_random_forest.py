from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import pandas as pd

data = pd.read_csv('datasets/GOOG_max.csv')

#data.drop(data.tail(3).index, inplace=True)

# set the index to be the date
#data.set_index(data.Date, inplace=True)
#data.drop('Date', 1, inplace=True)
data.index = data['Date']
data.sort_index(ascending=True, axis=0)

# to use this, comment out the train_test_split library
TOTAL = data.count()[0]
TEST = 10       # how many entries to use to test the model
TRAIN = TOTAL - TEST

features = ['Open', 'High', 'Low']
X = data[features]
y = data['Close']

X_train, X_test = X[:TRAIN], X[TRAIN:]
y_train, y_test = y[:TRAIN], y[TRAIN:]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# model = RandomForestRegressor(n_estimators=200, bootstrap=True, min_samples_leaf=25)

model = RandomForestRegressor(n_estimators=200, random_state=0)
model.fit(X_train, y_train)

# predict
prediction = model.predict(X_test)

print(explained_variance_score(y_test, prediction))
print(prediction[-10:])
print(y_test)

