import pandas as pd

data_set = pd.read_csv('/Users/bersus96/Desktop/T2_cleaned.csv', 
                       sep=',')

# Principal Component Analysis
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
data_norm = scaler.fit_transform(data_set)

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
