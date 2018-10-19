from common_clustering import CommonClustering

clustering_features = CommonClustering(r'C:\Users\ivangarrera\Desktop\T2_cleaned.csv')

clustering_features.PrincipalComponentAnalysis(num_components=2)
clustering_features.PlotPCAResult()

# Get the number of clusters that provides the best results
ideal_number_of_clusters = clustering_features.getBestNumberOfClusters()

# Plot silhuettes array
clustering_features.PlotSilhouettes()

# Print k-means with the best number of clusters that have been found
clustering_features.KMeansWithIdeal(ideal_number_of_clusters)

