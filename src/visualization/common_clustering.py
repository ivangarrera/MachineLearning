# -*- coding: utf-8 -*-

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import pandas as pd


class CommonClustering:
    def __init__(self, data_path):
        self.data_set = pd.read_csv(data_path, sep=',')
        self.reduced_data = None
        self.silhouettes = []
        self.attr = ''
    
    def PrincipalComponentAnalysis(self, num_components, plot_graph=True):
        scaler = preprocessing.MinMaxScaler()
        data_norm = scaler.fit_transform(self.data_set)
        
        estimator = PCA(n_components=num_components)
        self.reduced_data = estimator.fit_transform(data_norm)
        print(estimator.explained_variance_ratio_)
        
        if plot_graph:
            plt.scatter(self.reduced_data[:,0], self.reduced_data[:,1])
            plt.savefig(r'../../reports/figures/PCA_{}.png'.format(self.attr))
            plt.show()
        

    def getBestNumberOfClusters(self):
        #    -------------- Silhuette cofficient -----------------
        #    Best value is 1 and worst value is -1
        #    Value near 0 indicates overlapping clusters
        #    Negative value indicates that a sample has been asigned to the wrong cluster
        #    Value near 1 indicates that the object is well matched with its cluster    
                
        # Find the number of clusters that gives the best silhouettes coefficient
        for clus in range(2, 50):
            labels = KMeans(n_clusters=clus, init="random", random_state=0).\
                            fit_predict(self.reduced_data)
            self.silhouettes.append(metrics.silhouette_score(self.reduced_data, labels))
            
            if max(self.silhouettes) == self.silhouettes[len(self.silhouettes) - 1]:
                ideal_number_of_clusters = clus
            
        print (max(self.silhouettes),'- Cluster:', ideal_number_of_clusters)
        return ideal_number_of_clusters
    
    def KMeansWithIdeal(self, number_of_clusters, plot_graph=True):
        k_means = KMeans(n_clusters=number_of_clusters, init="random", random_state=0)
        labels = k_means.fit_predict(self.reduced_data)
        
        if plot_graph:
            plt.scatter(self.reduced_data[:,0], self.reduced_data[:,1], c=labels)
            # Plotting centroids
            plt.scatter(k_means.cluster_centers_[:,0], k_means.cluster_centers_[:,1],
                        c='red', s=50)
            plt.savefig(r'../../reports/figures/KMeans_{}.png'.format(self.attr))
            plt.show()
        return labels
    
    def AgglomerativeClusteringWithNearestNeighbors(self, plot_graph=True, nneighbors=3, nclusters=5):
        # Clustering (Agglomerative Clustering)
        # KNN gives a connectivity matrix which will be used by Agglomerative Clustering. 
        # This clustering algorithm will create a hierarchy with all points connected the ones
        # with the others, so all points in the connectivity matrix must be connected.
        # knn_graph returns a matrix with elements connected. Element a is connected to 
        # element b if matrix[a][b] = 1. 
        
        knn_graph = kneighbors_graph(self.reduced_data, n_neighbors=nneighbors, include_self=False)
        
        
        labels =  AgglomerativeClustering(n_clusters=nclusters, connectivity=knn_graph, 
                                          linkage="complete").fit_predict(self.reduced_data)
        
        if plot_graph:
            plt.scatter(self.reduced_data[:,0], self.reduced_data[:,1], c=labels)
            plt.savefig(r'../../reports/figures/AgglomerativeClustering_{}.png'.format(self.attr))
            plt.show()
    
    def DBSCANClustering(self, plot_graph=True, eps=0.1, min_samples=5):
        # 2. Execute clustering (DBSCAN)
        # eps=Neighborhood: distance radio to search neighbor points
        # A core point is a point which has at least 'minPts' points in its neighborhood
        # A border point is a point which has less than minPts in its neighborhood but
        # it has a core point in its neighborhood
        # A noise point doesn't have a core point in its neighborhood neither minPts points in its neighborhood
        labels = DBSCAN(eps=eps, min_samples=min_samples).\
                                        fit_predict(self.reduced_data)
                                        
        if plot_graph:
            plt.scatter(self.reduced_data[:,0], self.reduced_data[:,1], c=labels)
            plt.savefig(r'../../reports/figures/DBSCAN_{}.png'.format(self.attr))
            plt.show()
        return labels
    
    def PlotSilhouettes(self):
        plt.plot(self.silhouettes, 'bx-')
        plt.savefig(r'../../reports/figures/Silhouettes_{}.png'.format(self.attr))
        plt.show()
        
    
