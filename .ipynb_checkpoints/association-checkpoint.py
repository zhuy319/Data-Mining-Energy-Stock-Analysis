import sqlite3
import pandas as pd
import warnings
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import seaborn as sns
import matplotlib.pyplot as plt

#create transaction list for association analysis
#group stocks by dates
def transactions(dataset):
    return dataset

#encode the dataset to true and false
def encoder(transactions_list):
    te = TransactionEncoder()
    te_ary = te.fit(transactions_list).transform(transactions_list)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return dataset
 
#create frequent itemset with support value
def frequent(dataset, support):
    frequent_itemsets = apriori(df, min_support=support, use_colnames=True)
    return frequent_itemsets

#develop association_rules
def rules(frequent_itemsets,metrics,threshhold):
    rules=association_rules(frequent_itemsets, metric="metrics", min_threshold=threshhold)
    return rules

#select rules that interesting enough to study
def interestness(rules, treshhold):
    return newrules