# Stock-Price-Forecasting-using-Machine-Learning
This project focuses on predicting the stock prices of Apple Inc. (AAPL) using a Stacked Long Short-Term Memory (LSTM) model. The approach involves several key steps:

Data Collection: Historical stock price data for Apple Inc. is fetched using the yfinance library and saved to a CSV file.

Data Preprocessing: The data is normalized using MinMaxScaler, and then split into training and testing sets. Sequences are created to serve as input for the LSTM model.

Model Building: A Stacked LSTM model is constructed using TensorFlow and Keras. The model consists of three LSTM layers, each with 50 units, followed by a Dense layer with a single unit for output.

Training: The model is trained on the prepared dataset for 100 epochs with a batch size of 64, using mean squared error as the loss function and the Adam optimizer.

Prediction: The trained model is used to predict stock prices on the test data, and also to forecast the next 30 days of stock prices.

Visualization: The results, including actual vs. predicted stock prices and future forecasts, are visualized using Matplotlib to assess the model's performance.

The project demonstrates the application of deep learning techniques to time series forecasting in the context of financial data.
