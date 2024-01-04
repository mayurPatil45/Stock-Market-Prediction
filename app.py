import numpy as np
import pandas as pd
import quandl
import datetime
import matplotlib.pyplot as plt


start_date = datetime.date(2009, 3, 8)
end_date = datetime.date.today()
data = quandl.get('FSE/SAP_X', start_date=start_date, end_date=end_date)
data.to_csv('stock_market.csv')
df = pd.DataFrame(data, columns=['Close'])
df = df.reset_index()

import matplotlib.dates as mdates

years = mdates.YearLocator()
yearsFmt = mdates.DateFormatter('%Y')
fig, ax = plt.subplots()
ax.plot(df['Date'], df['Close'])
# Format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)

plt.title('Close Stock Price History [2009 - 2019]')
plt.xlabel('Date')
plt.ylabel('Closing Stock Price in $')
plt.show()

# Import package for splitting data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.20, random_state=0)

# Reshape index column to 2D array for .fit() method
X_train = np.array(train.index).reshape(-1, 1)
y_train = train['Close']

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Train set graph
plt.title('Linear Regression | Price vs Time')
plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Create test arrays
X_test = np.array(test.index).reshape(-1, 1)
y_test = test['Close']

# Generate array with predicted values
y_pred = model.predict(X_test)
print("predicted stock market price of year are:", model.predict([[2026]]))
print("predicted stock market price of year are:", model.predict([[2052]]))
