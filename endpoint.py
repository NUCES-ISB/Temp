
from flask import Flask, request, jsonify,render_template
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

app = Flask(_name_,template_folder='/app')

data = pd.read_csv('data.csv')

# Split the data into training and testing sets (80% training, 20% testing)
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Define the input features (Open, High, Low, and Volume)
X_train = train_data[['Open','High', 'Low', 'Volume']]
X_test = test_data[['Open','High', 'Low', 'Volume']]


# Define the target variable (Close)
y_train = train_data['Close']
y_test = test_data['Close']

# Create polynomial features of degree 2
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the polynomial regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict the Close values for the test data
y_pred = model.predict(X_test_poly)

# Print the model's mean squared error on the test data
mse = mean_squared_error(y_test, y_pred)

joblib.dump(model ,'model.pkl')

model = joblib.load('model.pkl')

data = pd.read_csv("data.csv")

