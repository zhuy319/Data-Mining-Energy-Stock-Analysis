from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import BisectingKMeans
import numpy as np
import matplotlib.pyplot as plt

def sklearn_kmeans(dataset, k=3):
    clusterer = KMeans(n_clusters=k, random_state=0).fit(dataset)
    silhouette_coeff = silhouette_score(dataset, clusterer.labels_, metric='euclidean')

    return clusterer.labels_, silhouette_coeff

def minib_kmeans(dataset, k):
    clusterer = MiniBatchKMeans(n_clusters=k, init = 'k-means++', random_state=0).fit(dataset)
    silhouette_coeff = silhouette_score(dataset, clusterer.labels_, metric='euclidean')
    
    return clusterer.labels_, silhouette_coeff


def bisec_kmeans(dataset, k):
    clusterer = BisectingKMeans(n_clusters=k, init = 'k-means++', random_state=0, bisecting_strategy='largest_cluster').fit(dataset)
    silhouette_coeff = silhouette_score(dataset, clusterer.labels_, metric='euclidean')
    
    return clusterer.labels_, silhouette_coeff
 

    
