import pandas as pd
path_to_data = r'C:\Users\ivangarrera\Desktop\T4.csv'
path_to_moriarty = r'C:\Users\Ivangarrera\Downloads\Moriarty.csv'

data_set = pd.read_csv(path_to_data, sep=',')
data_set_moriarty = pd.read_csv(path_to_moriarty, sep=',', warn_bad_lines=False, error_bad_lines=False)

characteristics = ["UUID", "Total_CPU", "TotalMemory_used_size", "Traffic_TotalRxBytes", "Traffic_TotalTxBytes", "Traffic_TotalWifiRxBytes", "Traffic_TotalWifiTxBytes", "procs_running", "connectedWifi_SSID"]
data_set = data_set.loc[:, data_set.columns.str.contains('|'.join(characteristics))]
        
for index in range(len(data_set_moriarty["UUID"])):
    df = data_set.query("UUID <= " + str(data_set_moriarty["UUID"][index]) +" <= UUID + 5000")
    if not df.empty:
        data_set_moriarty["UUID"][index] = df["UUID"].values[0]

df = pd.merge(data_set, data_set_moriarty, on="UUID")

# Remove action and details columns
X = df[["Total_CPU", "TotalMemory_used_size", "Traffic_TotalRxBytes", "Traffic_TotalTxBytes", "Traffic_TotalWifiRxBytes", "Traffic_TotalWifiTxBytes", "procs_running"]]

target = []
for i in range(len(df["ActionType"])):
    if "malicious" in df["ActionType"][i]:
        target.append(1)
    else:
        target.append(0)

######################################################
######### NORMALIZE AND SELECT TRAIN AND TEST ########
######################################################

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, target, test_size=0.4)

######################################################
############### NAIVE BAYES ##########################
######################################################

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, Y_train)

predicion = model.predict(X_test)   

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(Y_test, predicion)
print("Error measure {}".format(mae))

import numpy as np
import matplotlib.pyplot as plt

xx = np.stack(i for i in range(len(Y_test)))
plt.scatter(xx, Y_test, c='r', label='data')
plt.plot(xx, predicion, c='g', label='prediction')
plt.axis('tight')
plt.legend()
plt.title("Gaussian NaiveBayes")
plt.savefig(r'../../reports/figures/NaiveBayesPrediction.png')
plt.show()

######################################################
################ k-NEAREST NEIGHBORS #################
######################################################

xx = np.stack(i for i in range(len(target)))

from sklearn import neighbors
from sklearn.cross_validation import cross_val_score

for i, weights in enumerate(['uniform', 'distance']):
    total_scores = []
    for n_neighbors in range(1,30):
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        knn.fit(X_scaled, target)
        scores = -cross_val_score(knn, X_scaled, target, scoring="neg_mean_absolute_error", cv=10)
        total_scores.append(scores.mean())
    
    plt.plot(range(0, len(total_scores)), total_scores, marker='x', label=weights)
    plt.ylabel('cv_score')
plt.legend()
plt.show()

# Prediction
n_neighbors = 4
for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_pred = knn.fit(X_scaled, target).predict(X_scaled)
    
    plt.subplot(2,1,i+1)
    plt.plot(xx, target, c='k', label='data')
    plt.plot(xx, y_pred, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title('Hola')
plt.show()
    