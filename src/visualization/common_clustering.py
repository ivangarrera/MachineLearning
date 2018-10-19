# -*- coding: utf-8 -*-

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd


class CommonClustering:
    def __init__(self, data_path):
        self.data_set = pd.read_csv(data_path, sep=',')
        self.reduced_data = None
        self.silhouettes = []
    
    def PrincipalComponentAnalysis(self, num_components):
        scaler = preprocessing.MinMaxScaler()
        data_norm = scaler.fit_transform(self.data_set)
        
        self.reduced_data = PCA(n_components=num_components).fit_transform(data_norm)
        
        
    def PlotPCAResult(self):
        x = self.reduced_data[:,0]
        y = self.reduced_data[:,1]
        plt.scatter(x,y)
        plt.show()

    def getBestNumberOfClusters(self):
        """
            -------------- Silhuette cofficient -----------------
            Best value is 1 and worst value is -1
            Value near 0 indicates overlapping clusters
            Negative value indicates that a sample has been asigned to the wrong cluster
            Value near 1 indicates that the object is well matched with its cluster    
        """
                
        # Find the number of clusters that gives the best silhouettes coefficient
        for clus in range(2, 50):
            labels = KMeans(n_clusters=clus, init="random", random_state=0).\
                            fit_predict(self.reduced_data)
            self.silhouettes.append(metrics.silhouette_score(self.reduced_data, labels))
            
            if max(self.silhouettes) == self.silhouettes[len(self.silhouettes) - 1]:
                ideal_number_of_clusters = clus
            
        print (max(self.silhouettes),'- Cluster:', ideal_number_of_clusters)
        return ideal_number_of_clusters
    
    def KMeansWithIdeal(self, number_of_clusters):
        k_means = KMeans(n_clusters=number_of_clusters, init="random", random_state=0)
        labels = k_means.fit_predict(self.reduced_data)
        
        x = self.reduced_data[:,0]
        y = self.reduced_data[:,1]
        plt.scatter(x, y, c=labels)
        # Plotting centroids
        plt.scatter(k_means.cluster_centers_[:,0], k_means.cluster_centers_[:,1], c='red',s=50)
        plt.show()
    
    def PlotSilhouettes(self):
        plt.plot(self.silhouettes, 'bx-')
        plt.show()
        
    
