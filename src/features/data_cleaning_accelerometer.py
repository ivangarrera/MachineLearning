import pandas as pd

data_set = pd.read_csv('D:\Ing. Informatica\Cuarto\Machine Learning\T2.csv', 
                       sep=',')
data_set = data_set[(data_set['TimeStemp'] > '2016-04-28 00:00:00') & (data_set['TimeStemp'] <= '2016-05-02 23:59:59')]
# Remove strings columns
for column in data_set:
    if type(data_set[column][0]) is str:
        data_set = data_set.drop(column, axis=1)
        
# Remove columns with all null values
for column in data_set:
    if data_set[column].isnull().all():
        data_set = data_set.drop(column, axis=1)

# Remove duplicate rows
data_set = data_set.drop_duplicates()  
# Remove rows with any null value
data_set = data_set.dropna()

# Remove version, because 2.3.3 is not a valid float value and PCA could fail
data_set = data_set.drop('Version', axis=1)

data_set_mean = data_set [[column for column in data_set if "Accelerometer" in column and column.endswith("MEAN")]]

data_set_mean.to_csv('D:\Ing. Informatica\Cuarto\Machine Learning\T2_cleaned_accelerometer.csv', 
                sep=',', index=False)
