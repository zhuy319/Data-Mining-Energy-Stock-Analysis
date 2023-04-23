# Author: Natalia Khuri
# Date Created: September 1, 2022
# Updated: September 18, 2022

import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_regression

def remove_miss_columns(dataset):
    columns_to_drop = set()
    for col in dataset.columns:
        if (dataset[col].isna().sum())>dataset.shape[0]*0.5:
            columns_to_drop.add(col)
    dataset = dataset.drop(columns=columns_to_drop)
    return dataset

def remove_outliers(dataset, num_std=3):
   numeric_columns = dataset.select_dtypes('float64').columns
   for column in numeric_columns: 
      mean = dataset[column].mean()
      sd = dataset[column].std() 
      dataset = dataset[(dataset[column] <= mean + (num_std * sd))]
      dataset = dataset[(dataset[column] >= mean - (num_std * sd))]
   return dataset

def remove_features_near_zero_variance(dataset, threshold=1e-4):
    numeric_columns = dataset.select_dtypes('float64').columns
    columns_to_drop = []
    for column in numeric_columns:
        if dataset[column].std()**2 < threshold:
            columns_to_drop.append(column)
    dataset = dataset.drop(columns=columns_to_drop)
    return dataset

def remove_highly_correlated_features(dataset, threshold=0.9):
    numeric_columns = dataset.select_dtypes('float64').columns
    correlation_matrix = dataset[numeric_columns].corr().abs()

    columns_to_drop = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
               colname = correlation_matrix.columns[i]
               columns_to_drop.add(colname)

    dataset = dataset.drop(columns=columns_to_drop)
    return dataset
    
def basic_features_selection(dataset):
    dataset = remove_features_near_zero_variance(dataset, threshold=1e-4)
    dataset = remove_highly_correlated_features(dataset, threshold=0.9)
    return dataset

def k_features_selection(dataset, X_clf, Y_clf,k):
    dataset = basic_features_selection(dataset)
    dataset = k_best_selection(dataset, X_clf, Y_clf,k)
    return dataset

def encode_features(dataset):
    category_columns = dataset.select_dtypes('category').columns
    columns_to_drop = set()
    for col in category_columns:
        if dataset[col].nunique()>dataset.shape[0]*0.017:
            columns_to_drop.add(col)
    dataset = dataset.drop(columns=columns_to_drop)
    category_columns = dataset.select_dtypes('category').columns
    dataset = pd.get_dummies(dataset, columns=category_columns)
    return dataset
    

###############
def k_best_selection(dataset, X_clf, Y_clf,k):
    np.seterr(invalid='ignore')
    selector = SelectKBest(f_regression, k)
    selector.fit(X_clf, Y_clf)
    cols = selector.get_support(indices=True)
    dataset = dataset.iloc[:,cols]
    dataset.insert(0, X_clf, Y_clf)
    return dataset

def run_linear_regression_tests(dataset):
    numeric_columns = dataset.select_dtypes('float64').columns
    for column in numeric_columns:
        target = dataset[column].to_numpy()
        features = dataset[numeric_columns].drop([column], axis=1)
        features = np.array(features)
        regression_model = LinearRegression().fit(features, target)
        print('Target: ' + column + 
              '\tFitted R2: ' + str(round(regression_model.score(features, target), 2)))

def check_datasets(dir_path, index_col, transpose_flag):
   file_count = 0
   for filename in os.listdir(dir_path):
      if filename.startswith('.'):
         continue
      file = os.path.join(dir_path, filename)
      if os.path.isfile(file):
         if index_col != None:
            dataset = pd.read_csv(file, index_col=index_col, encoding='utf-8')
         else:
            dataset = pd.read_csv(file)
         if transpose_flag:
            dataset = dataset.T
         file_count = file_count + 1

         if file_count == 1:
            expected_nrows = dataset.shape[0]
            expected_ncols = dataset.shape[1]
            row_names = dataset.index
            col_names = dataset.columns

         if dataset.shape[0] != expected_nrows:
            print('Number of rows do not match for ' + file)
         if dataset.shape[1] != expected_ncols:
            print('Number of cols do not match for ' + file)
         if dataset.isnull().values.any() == True:
            print('Missing values in ' + file)
         if dataset.duplicated().any():
            print('Duplicated rows in ' + file)
         if list(row_names) != list(dataset.index):
            print('Rows differ in ' + file)
         if list(col_names) != list(dataset.columns):
            print('Columns differ in ' + file)

   print('Total number of files ' + str(file_count))    
