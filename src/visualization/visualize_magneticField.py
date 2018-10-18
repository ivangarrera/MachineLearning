import pandas as pd

data_set = pd.read_csv(r'C:\Users\ivangarrera\Desktop\T2_cleaned.csv',
                       sep=',')

data_set = data_set.head(5000)

# Principal Component Analysis
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
data_norm = scaler.fit_transform(data_set)

from sklearn.decomposition import PCA
reduced_data = PCA(n_components=2).fit_transform(data_norm)

import matplotlib.pyplot as plt


x = reduced_data[:,0]
y = reduced_data[:,1]
plt.scatter(x,y)
plt.show()

# Clustering
# K means algorithm
from sklearn.cluster import KMeans
from sklearn import metrics

"""
    -------------- Silhuette cofficient -----------------
    Best value is 1 and worst value is -1
    Value near 0 indicates overlapping clusters
    Negative value indicates that a sample has been asigned to the wrong cluster
    Value near 1 indicates that the object is well matched with its cluster    
"""

ideal_number_of_clusters = 1000000
silhouettes = []

# Find the number of clusters that gives the best silhouettes coefficient
for clus in range(2, 50):
    labels = KMeans(n_clusters=clus, init="random", random_state=0).fit_predict(reduced_data)
    silhouettes.append(metrics.silhouette_score(reduced_data, labels))
    if max(silhouettes) == silhouettes[len(silhouettes) - 1]:
        ideal_number_of_clusters = clus

plt.plot(silhouettes)
plt.show()
print("Best silhouette value: {}\nIdeal number of clusters: {}\n".\
      format(max(silhouettes), ideal_number_of_clusters))

# Print k-means with the best number of clusters that have been found
k_means = KMeans(n_clusters=ideal_number_of_clusters, init="random", random_state=0)
labels = k_means.fit_predict(reduced_data)

plt.scatter(x, y, c=labels)
# Plotting centroids
plt.scatter(k_means.cluster_centers_[:,0], k_means.cluster_centers_[:,1], c='red',s=50)
plt.show()

