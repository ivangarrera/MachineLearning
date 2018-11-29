from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.cross_validation import cross_val_score

class SupervisedLearning:
    def __init__(self, merged, target):
        self.merged = merged
        self.target = target
        self.X_Train = None
        self.Y_Train = None
        self.X_Test = None
        self.Y_Test = None
        self.X_scaled = None
        
    def NormalizeData(self):
        scaler = preprocessing.MinMaxScaler().fit(self.merged)
        self.X_scaled = scaler.transform(self.merged)
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
        train_test_split(self.X_scaled, self.target, test_size=0.4)
    
    def NaiveBayesAlgorithm(self):
        model = GaussianNB()
        model.fit(self.X_Train, self.Y_Train)
        
        predicion = model.predict(self.X_Test)
        mae = mean_absolute_error(self.Y_Test, predicion)
        print("Error measure {}".format(mae))
        
        xx = np.stack(i for i in range(len(self.Y_Test)))
        plt.scatter(xx, self.Y_Test, c='r', label='data')
        plt.plot(xx, predicion, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("Gaussian NaiveBayes")
        plt.savefig(r'../../reports/figures/NaiveBayesPrediction.png')
        plt.show()
        
    def KNearestNeighborsAlgorithm(self):
        xx = np.stack(i for i in range(len(self.target)))
        for i, weights in enumerate(['uniform', 'distance']):
            total_scores = []
            for n_neighbors in range(1,30):
                knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
                knn.fit(self.X_scaled, self.target)
                scores = -cross_val_score(knn, self.X_scaled, self.target,
                                          scoring="neg_mean_absolute_error", cv=10)
                total_scores.append(scores.mean())
            
            plt.plot(range(0, len(total_scores)), total_scores, marker='x', label=weights)
            plt.ylabel('cv_score')
        plt.legend()
        plt.show()
        return xx
    
    def MakePrediction(self, xx):
        n_neighbors = 4
        for i, weights in enumerate(['uniform', 'distance']):
            knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            y_pred = knn.fit(self.X_scaled, self.target).predict(self.X_scaled)
            
            plt.subplot(2, 1, i + 1)
            plt.plot(xx, self.target, c='k', label='data')
            plt.plot(xx, y_pred, c='g', label='prediction')
            plt.axis('tight')
            plt.legend()
            plt.title('Prediction')
        plt.show()