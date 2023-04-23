import numpy as np
from numpy import *
import pandas as pd
import preprocessor
import warnings
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

# calculate naive MAE
def calculate_naive(y_train):
    pred_train = [mean(y_train)] * len(y_train)
    return mean_absolute_error(y_train, pred_train)

global prediction_preprocessor
# cross validation to avoid overfitting
def cross_validation(data):
    prediction_preprocessor = preprocessor.Preprocessor(0)
    splits = [
        "Blocked Time Series Split",
        "Time Series Split"
    ]
    names = [
        "Linear Regression",
        "Support Vector Machines",
        "Decision Tree",
        "Random Forest"
    ]
    classifiers = [
        LinearRegression(),
        svm.SVR(),
        DecisionTreeRegressor(random_state = random.seed(9)),
        RandomForestRegressor(random_state = random.seed(9))
    ]
    train_MAEs = [[0,0,0,0], [0,0,0,0]]
    test_MAEs = [[0,0,0,0], [0,0,0,0]]
    train_naive = [0,0]
    
    num_train = 60 # Increment of how many starting points
    len_train = 35 # Length of each train-test set
    fold_id = (data.shape[0]-num_train)//num_train
   
    for i in range(0,fold_id):
        # run cross validation for blocked split with 20-day price as training set
        df0 = data.iloc[i * num_train : (i * num_train) + len_train]
        X = df0[['Date_Year', 'Date_Month', 'Date_Day', 'Date_WeekDay']]
        y = df0['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=15, shuffle=False)
        train_naive[0] += calculate_naive(y_train)
                
        # preprocess training subset
        prediction_preprocessor.fit(X_train)
        X_train = prediction_preprocessor.transform(X_train)

        # apply the same preprocessing parameters to test subset
        X_test = prediction_preprocessor.transform(X_test)

        # train models on train data and test on test data
        for j in range(0, len(classifiers)):
            classifiers[j].fit(X_train, y_train)
            pred_train = classifiers[j].predict(X_train)
            train_MAEs[0][j] += mean_absolute_error(y_train,pred_train)
            pred_test = classifiers[j].predict(X_test)
            test_MAEs[0][j] += mean_absolute_error(y_test,pred_test)
            
        # run cross validation for time series split with growing length of training set
        df1 = data.iloc[i * num_train : (i * num_train) + len_train]
        X = df1[['Date_Year', 'Date_Month', 'Date_Day', 'Date_WeekDay']]
        y = df1['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=15, shuffle=False)
        train_naive[1] += calculate_naive(y_train)
        
        # preprocess training subset
        prediction_preprocessor.fit(X_train)
        X_train = prediction_preprocessor.transform(X_train)

        # apply the same preprocessing parameters to test subset
        X_test = prediction_preprocessor.transform(X_test)

        # train models on train data and test on test data
        for k in range(0, len(classifiers)):
            classifiers[k].fit(X_train, y_train)
            pred_train = classifiers[j].predict(X_train)
            train_MAEs[1][k] += mean_absolute_error(y_train,pred_train)
            pred_test = classifiers[j].predict(X_test)
            test_MAEs[1][k] += mean_absolute_error(y_test,pred_test)

    # compute averaged accuracy for each model
    for a in range(0,2):
        train_naive[a] = train_naive[a]/fold_id
        for b in range(0, len(classifiers)):
            train_MAEs[a][b] = train_MAEs[a][b]/fold_id
            test_MAEs[a][b] = test_MAEs[a][b]/fold_id
    
    metrics = pd.DataFrame(test_MAEs)
    metrics.index = splits
    metrics.columns = names
    print('Test MAEs:')
    print(metrics)
    
    min_index = np.argwhere(test_MAEs == np.min(test_MAEs))[0]
    fitting = [train_naive[min_index[0]], train_MAEs[min_index[0]][min_index[1]], test_MAEs[min_index[0]][min_index[1]]]
    print('The best model is ' + names[min_index[1]] + ' using ' + splits[min_index[0]] +' approach!')
    
    
    # create prediction model based on the best performed model
    best_model = classifiers[min_index[1]]

    if min_index[0] == 0:
        new_train = data.tail(20)
    else:
        new_train = data
    
    features = new_train[['Date_Year', 'Date_Month', 'Date_Day', 'Date_WeekDay']]
    target = new_train['Close']
    prediction_preprocessor.fit(features)
    features = prediction_preprocessor.transform(features)
    best_model.fit(features, target)
        
    return fitting, names[min_index[1]], splits[min_index[0]], best_model, prediction_preprocessor


def predict(test_data, preprocessor, model):
    X_test = test_data[['Date_Year', 'Date_Month', 'Date_Day', 'Date_WeekDay']]
    X_test = preprocessor.transform(X_test)
    test_pred = model.predict(X_test)
    return test_pred