from common_clustering import CommonClustering

#clustering_features = CommonClustering(r'C:\Users\ivangarrera\Desktop\T2_cleaned.csv')
clustering_features = CommonClustering('D:\Ing. Informatica\Cuarto\Machine Learning\T2_cleaned_magneticField.csv')

clustering_features.PrincipalComponentAnalysis(num_components=2)
clustering_features.PlotPCAResult()

# Get the number of clusters that provides the best results
ideal_number_of_clusters = clustering_features.getBestNumberOfClusters()

# Plot silhuettes array
clustering_features.PlotSilhouettes()

# Print k-means with the best number of clusters that have been found
clustering_features.KMeansWithIdeal(ideal_number_of_clusters)

def plotdata(data,labels,name): #def function plotdata
#colors = ['black']
    fig, ax = plt.subplots()
    plt.scatter(data[:,0], data[:,1], c=labels)
    ax.grid(True)
    fig.tight_layout()
    plt.title(name)
    plt.show()


# 1. setting parameters
# 1.1 Compute the similarity/distance matrix (high cost)
# The graphic could offer better results (improve it!!)
import matplotlib.pyplot as plt
import sklearn.neighbors

dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(clustering_features.reduced_data)

# 1.2 Compute the k-nearest neighboors
minPts=5
from sklearn.neighbors import kneighbors_graph
A = kneighbors_graph(clustering_features.reduced_data, minPts, include_self=False)
Ar = A.toarray()

seq = []
for i,s in enumerate(clustering_features.reduced_data):
    for j in range(len(clustering_features.reduced_data)):
        if Ar[i][j] != 0:
            seq.append(matsim[i][j])
            
seq.sort()
plt.plot(seq)
plt.show()

# 2. Execute clustering (dbscan)
import sklearn.cluster
labels = sklearn.cluster.DBSCAN(eps=0.08, min_samples=minPts).fit_predict(clustering_features.reduced_data)

# 3. Plot the results
plotdata(clustering_features.reduced_data,labels, 'dbscan')

