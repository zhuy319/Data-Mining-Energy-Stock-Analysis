import numpy as np
import pandas as pd
import warnings
import my_utils
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

# Preprocessor for prediction task
class Preprocessor:
    # initialize constructors
    def __init__(self):
        self.scaler = preprocessing.MinMaxScaler()
    
    # fit dataset and store constructor objects
    def fit(self, data):
        self.scaler = self.scaler.fit(data)
        
    # transform dataset with flags 
    def transform(self, data):
        data = data.copy()
        data = self.scaler.transform(data)
        return data
    
# calculate Test Metics MAE    
def calculate_MAE(y_test, prediction):
    sum = 0
    y_test = list(y_test)
    prediction = list(prediction)
    for i in range(0, len(y_test)):
        sum += abs(y_test[i] - prediction[i])
    MSE = sum/len(y_test)
    return MSE
    
# cross validation
# flag=0 indicate time series split, flag=1 indicate block time series split
# split_length is the number of dates accounted in the training set
def cross_validation(data, flag, split_length):
    preprocessor = Preprocessor()
    ModelNames = [
        "LSTM",
        "Decision Tree",
        "Random Forest",
        "Linear Regression"
        ]
    return bestModel

# prediction function to feed in the whole dataset and predict the future stock price
# flag=0 indicate time series split, flag=1 indicate block time series split
def prediction(train, test, flag):
    preprocessor = Preprocessor()
    print("Model Accuracy on real data:")
    plt.title("prediction plot")
    return