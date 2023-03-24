
from flask import Flask, request, jsonify,render_template
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
