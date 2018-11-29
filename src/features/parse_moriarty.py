from data_parsing_moriarty import ParsingMoriarty
from supervisedLearning import SupervisedLearning

path_to_data = r'C:\Users\ivangarrera\Desktop\T4.csv'
path_to_moriarty = r'C:\Users\Ivangarrera\Downloads\Moriarty.csv'

parsing_moriarty = ParsingMoriarty(path_to_data, path_to_moriarty)

characteristics = ["UUID", "Total_CPU", "TotalMemory_used_size", "Traffic_TotalRxBytes", 
                   "Traffic_TotalTxBytes", "Traffic_TotalWifiRxBytes", "Traffic_TotalWifiTxBytes", 
                   "procs_running", "connectedWifi_SSID"]
parsing_moriarty.MergeDataByUUID(characteristics)


# Remove action and details columns
characteristics = ["Total_CPU", "TotalMemory_used_size", "Traffic_TotalRxBytes", 
                   "Traffic_TotalTxBytes", "Traffic_TotalWifiRxBytes",
                   "Traffic_TotalWifiTxBytes", "procs_running"] 
parsing_moriarty.MergedWithNumericColumns(characteristics)

target = parsing_moriarty.GetTargets()

# Normalize data 
supervised_learning = SupervisedLearning(parsing_moriarty.merged_numeric, target)
supervised_learning.NormalizeData()

# NaiveBayes
supervised_learning.NaiveBayesAlgorithm()

# KNearest Neighbors
xx = supervised_learning.KNearestNeighborsAlgorithm()

# Make prediction
supervised_learning.MakePrediction(xx)


    