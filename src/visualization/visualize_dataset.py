from common_clustering import CommonClustering

#■clustering_features = CommonClustering(r'C:\Users\ivangarrera\Desktop\T2_cleaned.csv')
clustering_features = CommonClustering('D:\Ing. Informatica\Cuarto\Machine Learning\T2_cleaned_gyroscope.csv')

attr = list(clustering_features.data_set)[0][:list(clustering_features.data_set)[0].find('_')]
clustering_features.attr = attr

clustering_features.PrincipalComponentAnalysis(num_components=2)

# Get the number of clusters that provides the best results
ideal_number_of_clusters = clustering_features.getBestNumberOfClusters()

# Plot silhuettes array
clustering_features.PlotSilhouettes()

# Print k-means with the best number of clusters that have been found
labels = clustering_features.KMeansWithIdeal(ideal_number_of_clusters)

# Interprate k-means groups
clustering_features.data_set['labels'] = labels

data_set_labels_mean = clustering_features.data_set.groupby(['labels']).mean()

# Plot 3D graph to interpretate k-means groups
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data_set_labels_mean.values[:,0], 
           data_set_labels_mean.values[:,1], 
           data_set_labels_mean.values[:,2])
plt.savefig(r'../../reports/figures/centroids3D_{}.png'.format(attr))
plt.show()

# Agglomerative clustering algorithm using nearest neighbors matrix
clustering_features.AgglomerativeClusteringWithNearestNeighbors()

# DBSCAN Clustering algorithm
labels = clustering_features.DBSCANClustering()

# Interprate outliers
clustering_features.data_set['labels'] = labels
data_set_outliers = clustering_features.data_set.loc[(clustering_features.data_set['labels'] == -1)]

# Show outliers in a 3D graph with all points in the dataset
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(clustering_features.data_set.values[:,0], 
           clustering_features.data_set.values[:,1], 
           clustering_features.data_set.values[:,2])

ax.scatter(data_set_outliers.values[:,0], 
           data_set_outliers.values[:,1], 
           data_set_outliers.values[:,2], c='red', s=50)
plt.savefig(r'../../reports/figures/outliers3D_{}.png'.format(attr))
plt.show()

