import pandas as pd

data_set = pd.read_csv('/Users/bersus96/Desktop/T2_cleaned.csv', 
                       sep=',')

#data_set = data_set.head(20000)

# Principal Component Analysis
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
data_norm = scaler.fit_transform(data_set)

"""
for col in data_norm:
    for data in col:
        if data <0 or data >1: 
            print(data)
"""


from sklearn.decomposition import PCA
number_components = 2
estimator = PCA(number_components)
X_pca = estimator.fit_transform(data_norm)

print(estimator.explained_variance_ratio_)

import matplotlib.pyplot as plt
import numpy

x = X_pca[:,0]
y = X_pca[:,1]
plt.scatter(x,y)
plt.show()

ideal_number_of_cluster = 1000000
silhouettes = []

# Clustering
from sklearn.cluster import KMeans
from sklearn import metrics

for k in range(2,20):
    k_means = KMeans(n_clusters = k, init = "random",random_state=0)
    #k_means = KMeans(k, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = k_means.fit_predict(X_pca)
    silhouettes.append(metrics.silhouette_score(X_pca,labels))
    if max(silhouettes) == silhouettes[len(silhouettes) - 1]:
        ideal_number_of_cluster = k


# Print the Silouhette
plt.plot( silhouettes)
plt.xlabel('Number of clusters')
plt.ylabel('Silohouette')
plt.show()


print("Best silhouette value: {}\nIdeal number of cluster: {}\n ".\
      format(max(silhouettes),ideal_number_of_cluster))

k_means = KMeans(n_clusters = ideal_number_of_cluster, init = "random",random_state=0)
labels = k_means.fit_predict(X_pca)

plt.scatter(x,y, c=labels)
# plotting centroids
plt.scatter(k_means.cluster_centers_[:,0], k_means.cluster_centers_[:,1], c='red',s=50)
plt.show()
    