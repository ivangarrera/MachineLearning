from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import zero_one_loss
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier

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
        
        # 60% is training data and 40% is testing data
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = \
        train_test_split(self.X_scaled, self.target, test_size=0.4)
    
    def NaiveBayesAlgorithm(self):
        model = GaussianNB()
        model.fit(self.X_Train, self.Y_Train)
        
        predicion = model.predict(self.X_Test)
        zol = zero_one_loss(self.Y_Test, predicion)
        print("Error measure {}".format(zol))
        
        xx = np.stack(i for i in range(len(self.Y_Test)))
        plt.scatter(xx, self.Y_Test, c='r', label='data')
        plt.plot(xx, predicion, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("Gaussian NaiveBayes")
        plt.savefig(r'../../reports/figures/NaiveBayesPrediction.png')
        plt.show()
        
    def KNearestNeighborsAlgorithm(self):
        min_zol = 99999999
        best_neighbors = 0
        for n_neighbors in range(1,30):
            knn = neighbors.KNeighborsClassifier(n_neighbors)
            knn.fit(self.X_Train, self.Y_Train)
            prediction = knn.predict(self.X_Test)
            zol = zero_one_loss(self.Y_Test, prediction)
            if zol.item() < min_zol:
                min_zol = zol
                best_neighbors = n_neighbors
        
        # Knn with best neighbors
        knn = neighbors.KNeighborsClassifier(best_neighbors)
        knn.fit(self.X_Train, self.Y_Train)
        prediction = knn.predict(self.X_Test)
        print("Error measure {}".format(min_zol))
        
        xx = np.stack(i for i in range(len(self.Y_Test)))
        plt.scatter(xx, self.Y_Test, c='r', label='data')
        plt.plot(xx, prediction, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("KNearestNeighbors")
        plt.savefig(r'../../reports/figures/KNearestNeighborsPrediction.png')
        plt.show()
    
    def RandomForestAlgorithm(self):
        rf = RandomForestClassifier(n_estimators=1000, random_state=42)
        rf.fit(self.X_Train, self.Y_Train)
        prediction = rf.predict(self.X_Test)
        
        zol = zero_one_loss(self.Y_Test, prediction)
        print("Error measure {}".format(zol))
        
        xx = np.stack(i for i in range(len(self.Y_Test)))
        plt.scatter(xx, self.Y_Test, c='r', label='data')
        plt.plot(xx, prediction, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("RandomForestClassifier")
        plt.savefig(r'../../reports/figures/RandomForestClassifier.png')
        plt.show()

