from basic_data_cleaning import BasicCleaning

# Read csv and clean data
data_set = BasicCleaning.CleanData(path_to_data='D:\Ing. Informatica\Cuarto\Machine Learning\T2.csv')

data_set = data_set [[column for column in data_set if "Accelerometer" in column and column.endswith("MEAN")]]

data_set.to_csv('D:\Ing. Informatica\Cuarto\Machine Learning\T2_cleaned_accelerometer.csv', 
                sep=',', index=False)
