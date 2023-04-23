import pandas as pd
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from mlxtend.preprocessing import TransactionEncoder

class preprocessor():
    # initialize constructors
    def __init__(self, scaler_flag):
        # parameter for scale values
        # flag = 1 for MinMaxScaler which gives only positive scaling results
        # otherwise for StandardScaler which gives both negative and positive values 
        if scaler_flag==1:
            self.scaler = preprocessing.MinMaxScaler()
        else:
            self.scaler = preprocessing.StandardScaler()
        
    # fit dataset and store constructor objects
    def fit(self, data):
        self.selected_features = data.columns
        numeric_columns = data.select_dtypes('number').columns
        self.scaler = self.scaler.fit(data[numeric_columns])
        
    # transform dataset with flags 
    def transform(self, data):
        data = data.copy()
        data = self.scaler.transform(data)
        return data
    
    #encode the dataset to true and false
    def encoder(self, transactions_list):
        te = TransactionEncoder()
        te_ary = te.fit(transactions_list).transform(transactions_list)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        return df
