from data_parsing_moriarty import ParsingMoriarty
import pandas as pd
from supervisedLearning import SupervisedLearning

path_to_data = r'C:\Users\Ivangarrera\Desktop\T4.csv'
path_to_moriarty = r'C:\Users\Ivangarrera\Downloads\Moriarty.csv'

parsing_moriarty = ParsingMoriarty(path_to_data, path_to_moriarty)

characteristics = ["UUID", "Total_CPU", "TotalMemory_used_size", "Traffic_TotalRxBytes", 
                   "Traffic_TotalTxBytes", "Traffic_TotalWifiRxBytes", "Traffic_TotalWifiTxBytes", 
                   "procs_running", "connectedWifi_SSID"]
parsing_moriarty.MergeDataByUUID(characteristics)
x = parsing_moriarty.data_set
cuantosceros = 300
mydataset = pd.DataFrame(data=None, columns=parsing_moriarty.data_set.columns)

# Meter al dataset los no-virus
for i in range(300):
    index = int(i * len(parsing_moriarty.data_set) / cuantosceros)
    frames = [mydataset, parsing_moriarty.data_set.iloc[[index]]]
    mydataset = pd.concat(frames)

# Meter al dataset los viruses
for i in range(len(parsing_moriarty.data_set)):
    if parsing_moriarty.data_set["SessionType"][i] == 1:
        frames = [mydataset, parsing_moriarty.data_set.iloc[[i]]]
        mydataset = pd.concat(frames)

parsing_moriarty.merged = mydataset.sort_values(["UUID"])

# Remove action and details columns
characteristics = ["Total_CPU", "TotalMemory_used_size", "Traffic_TotalRxBytes", 
                   "Traffic_TotalTxBytes", "Traffic_TotalWifiRxBytes",
                   "Traffic_TotalWifiTxBytes", "procs_running"] 
parsing_moriarty.MergedWithNumericColumns(characteristics)

X = parsing_moriarty.merged_numeric
Y = parsing_moriarty.merged["SessionType"].values.tolist()

# Normalize data 
supervised_learning = SupervisedLearning(X, Y)
supervised_learning.NormalizeData()

# NaiveBayes
supervised_learning.NaiveBayesAlgorithm()

# KNearest Neighbors
supervised_learning.KNearestNeighborsAlgorithm()

# Random Forest
supervised_learning.RandomForestAlgorithm()