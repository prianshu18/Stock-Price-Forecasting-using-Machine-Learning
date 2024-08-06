# Stock Price Forecasting using Machine Learning
This project focuses on predicting the stock prices of Apple Inc. (AAPL) using a Stacked Long Short-Term Memory (LSTM) model. The approach involves several key steps:

Data Collection: Historical stock price data for Apple Inc. is fetched using the yfinance library and saved to a CSV file.

Data Preprocessing: The data is normalized using MinMaxScaler, and then split into training and testing sets. Sequences are created to serve as input for the LSTM model.

Model Building: A Stacked LSTM model is constructed using TensorFlow and Keras. The model consists of three LSTM layers, each with 50 units, followed by a Dense layer with a single unit for output.

Training: The model is trained on the prepared dataset for 100 epochs with a batch size of 64, using mean squared error as the loss function and the Adam optimizer.

Prediction: The trained model is used to predict stock prices on the test data, and also to forecast the next 30 days of stock prices.

Visualization: The results, including actual vs. predicted stock prices and future forecasts, are visualized using Matplotlib to assess the model's performance.

The project demonstrates the application of deep learning techniques to time series forecasting in the context of financial data.

# Description of Dataset

We are using the AAPL Dataset of stock prices derived from Tingo in Pandas Datareader Library using API Keys.
It contains of roughly 10,900 instances of data.
Number of features - 8, More can be derived
Features explanation: Date - Date of trading day.
Open - Opening price of stock on that day
High - Highest price of the stock on that day
Low - The lowest price of the stock on that day
Close - Price at which it closed at that day.
Volume - The total number of shares traded during the day.
Adj Close - Adjusted Closing Price adjusted for dividends and stock splits
Other derived features
Daily return - Self
Moving averages - The average closing price over a specific number of days. Smoothens out
short term fluctuations and helps identify trends
Rolling Volatility - 
Standard Deviation of the stock's returns over a specified window

RSI (Relative Strength Index) - A momentum oscillator that measures the speed and change of price movements

Some more basics of stocks 
Dividends are payments made by corporations to their shareholders, usually derived from profits. So it is basically Return on Investment.
DPS = Total div/ number of outstanding shares.
MCAP = Stock Price * Number of outstanding shares
PE ratio is the price to earning ratio, it is used to evaluate a company stock performance, price/earnings per share, so high pe ratio means higher growth or overvalued in future and low means undevalued or experiencing difficulties.

# Results

Please refer to the image below, it shows the actual prices versus the predicted prices of stock over time!


<img width="1028" alt="LSTM Results" src="https://github.com/user-attachments/assets/b414eb9c-1b0e-4234-a401-731b5d08b452">

