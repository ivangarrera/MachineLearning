import pandas as pd

data_set = pd.read_csv('D:\Ing. Informatica\Cuarto\Machine Learning\T2_cleaned_accelerometer.csv', 
                       sep=',')

# Principal Component Analysis
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
data_norm = scaler.fit_transform(data_set)

from sklearn.decomposition import PCA
from sklearn import metrics

number_components = 2
estimator = PCA(number_components)
data_reduction = estimator.fit_transform(data_norm)

print(estimator.explained_variance_ratio_)

import matplotlib.pyplot as plt

x = data_reduction[:,0]
y = data_reduction[:,1]
plt.scatter(x,y)
plt.show()

# Clustering
from sklearn.cluster import KMeans

iterations = 10
max_iter = 300 
tol = 1e-04 
random_state = 0
init = "random"
silhouettes = []
ideal_clusters = 1

for k in range(2,20):
    km = KMeans(k, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(data_reduction)
    silhouettes.append(metrics.silhouette_score(data_reduction, labels))
    if max(silhouettes) == silhouettes[len(silhouettes)-1]:
        ideal_clusters = k
    
# Plot the silhouettes values for diferent cluster
plt.plot(range(2,20), silhouettes, 'bx-')
plt.xlabel('k')
plt.ylabel('Coefficient')
plt.title('Silhouette Method showing the optimal k')
plt.show()


print (max(silhouettes),'- Cluster:', ideal_clusters)

# kmeans with the correct number of clusters
km = KMeans(ideal_clusters, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
labels = km.fit_predict(data_reduction)
plt.scatter(x,y, c = labels)

# plotting centroids
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='red',s=50)
plt.show()