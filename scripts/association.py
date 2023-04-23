import sqlite3
import pandas as pd
import warnings
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules
import seaborn as sns
import matplotlib.pyplot as plt
 
#create frequent itemset with support value
def frequent_apriori(dataset, support):
    frequent_itemsets = apriori(dataset, min_support=support, use_colnames=True)
    return frequent_itemsets

#create frequent itemset with support value
def frequent_fpmax(dataset, support):
    frequent_itemsets = fpmax(dataset, min_support=support, use_colnames=True)
    return frequent_itemsets

#create frequent itemset with support value
def frequent_fpgrowth(dataset, support):
    frequent_itemsets = fpgrowth(dataset, min_support=support, use_colnames=True)
    return frequent_itemsets

#develop association_rules
def confidence_rules(frequent_itemsets,threshhold):
    rules=association_rules(frequent_itemsets, metric="confidence", min_threshold=threshhold)
    return rules

#develop association_rules
def lift_rules(frequent_itemsets,threshhold):
    rules=association_rules(frequent_itemsets, metric="lift", min_threshold=threshhold)
    return rules