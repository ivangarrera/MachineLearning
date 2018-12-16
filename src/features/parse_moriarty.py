from data_parsing_moriarty import ParsingMoriarty
from supervisedLearning import SupervisedLearning

path_to_data = r'D:\Ing. Informatica\Cuarto\Machine Learning\T4.csv'
path_to_moriarty = r'D:\Ing. Informatica\Cuarto\Machine Learning\Moriarty.csv'

parsing_moriarty = ParsingMoriarty(path_to_data, path_to_moriarty)

characteristics = ["UUID", "Total_CPU", "TotalMemory_used_size", "Traffic_TotalRxBytes", 
                   "Traffic_TotalTxBytes", "Traffic_TotalWifiRxBytes", "Traffic_TotalWifiTxBytes", 
                   "procs_running", "connectedWifi_SSID"]

# Merge Sherlock and Moriarty datasets, using the UUID column. We consider that
# an attack 'a' (in the tx instant) has ocurred in the instant t1 if and only 
# if, t1 <= tx <= t1 + 5000
parsing_moriarty.MergeDataByUUID(characteristics)

# Create the dataset we are going to use to train and test the models
supervised_dataset = parsing_moriarty.CreateSupervisedDataset(number_of_non_attacks=300)
parsing_moriarty.merged = supervised_dataset.sort_values(["UUID"])

# Remove action and details columns, because these columns aren't numeric
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