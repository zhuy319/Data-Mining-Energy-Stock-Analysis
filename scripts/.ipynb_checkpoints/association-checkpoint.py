import sqlite3
import pandas as pd
import warnings
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import seaborn as sns
import matplotlib.pyplot as plt

#encode the dataset to true and false
def encoder(transactions_list):
    #association analysis
    te = TransactionEncoder()
    te_ary = te.fit(transactions_list).transform(transactions_list)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df
 
#create frequent itemset with support value
def frequent(dataset, support):
    frequent_itemsets = apriori(dataset, min_support=support, use_colnames=True)
    return frequent_itemsets

#develop association_rules
def crules(frequent_itemsets,threshhold):
    rules=association_rules(frequent_itemsets, metric="confidence", min_threshold=threshhold)
    return rules

#develop association_rules
def lrules(frequent_itemsets,threshhold):
    rules=association_rules(frequent_itemsets, metric="lift", min_threshold=threshhold)
    return rules