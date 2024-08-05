""" 

We are going to use Stacked LSTM to predict Stock Prices using Deep Learning.

We will use Pandas Library in Python to handle data.

We will transform stock price data into a range of 0 to 1 using min-max scaling.

Divide the dataset into 70% training data and 30% test data based on date

Data pre-processing involves converting data into independent and dependent features based on timestamps.

Data pre-processing involves splitting the dataset into training and test sets

Reshaping X train into a three-dimensional array is necessary before implementing Stacked LSTM.

Creating a stacked LSTM model for stock price prediction using sequential dense LSTM layers

The model's performance is evaluated using mean squared error (MSE).


"""

# Importing Libraries and installing dependencies.

import pandas_datareader as pdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

!pip install pandas-datareader

import yfinance as yf
import pandas as pd
# Fetch Apple stock data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="max")

# Save to CSV
df.to_csv('AAPL.csv')

# Display the first few rows
print(df.head())

print(df.shape[0])
!pip install yfinance
df=pd.read_csv('AAPL.csv')
df.head()



# Display basic information
print(df.info())

# Show the first few rows
print(df.head())

# Display summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())



# Plot closing price over time
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'])
plt.title('AAPL Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

# Calculate and plot daily returns
df['Daily Return'] = df['Close'].pct_change()
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Daily Return'])
plt.title('AAPL Stock Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.show()


# Histogram of daily returns
plt.figure(figsize=(10,6))
df['Daily Return'].hist(bins=50)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()

# Q-Q plot
from scipy import stats
fig, ax = plt.subplots(figsize=(10,6))
stats.probplot(df['Daily Return'].dropna(), dist="norm", plot=ax)
ax.set_title("Q-Q plot of Daily Returns")
plt.show()


# Calculate and plot rolling standard deviation
df['Volatility'] = df['Daily Return'].rolling(window=21).std() * np.sqrt(252)
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Volatility'])
plt.title('AAPL Stock 21-Day Rolling Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.show()

# Plot volume over time
plt.figure(figsize=(12,6))
plt.bar(df.index, df['Volume'])
plt.title('AAPL Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()

# Scatter plot of volume vs price
plt.figure(figsize=(10,6))
plt.scatter(df['Volume'], df['Close'])
plt.title('Volume vs Closing Price')
plt.xlabel('Volume')
plt.ylabel('Closing Price')
plt.show()


# Correlation matrix
corr_matrix = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of AAPL Stock Features')
plt.show()



# Calculate and plot 50-day and 200-day moving averages
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Close')
plt.plot(df.index, df['MA50'], label='50-day MA')
plt.plot(df.index, df['MA200'], label='200-day MA')
plt.title('AAPL Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


df.head()
df1=df.reset_index()['Close']
df1
plt.plot(df1)



from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

training_size,test_size

train_data


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
print(X_train.shape), print(y_train.shape)



print(X_test.shape), print(ytest.shape)


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

!pip install tensorflow

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


model.summary()

model.summary()

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

import tensorflow as tf

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))

### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


len(test_data)

x_input=test_data[341:].reshape(1,-1)
x_input.shape

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

temp_input


# demonstrate prediction for next 30 days
from numpy import array

lst_output=[]
n_steps=100
i=0

while(i<30):
    if(len(temp_input)>100):
        x_input = np.array(temp_input[-100:])  # Take only the last 100 elements
        print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.append(yhat[0,0])  # Append only the scalar prediction
        temp_input = temp_input[1:]  # Remove the oldest element
        lst_output.append(yhat[0,0])  # Append only the scalar prediction
        i=i+1
    else:
        x_input = np.array(temp_input)
        x_input = x_input.reshape((1, len(x_input), 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.append(yhat[0,0])  # Append only the scalar prediction
        print(len(temp_input))
        lst_output.append(yhat[0,0])  # Append o


day_new=np.arange(1,101)
day_pred=np.arange(101,131)

# Assuming you want to plot only the first 100 data points
plt.plot(day_new, scaler.inverse_transform(df1[1158:1158+100]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))
plt.show()
