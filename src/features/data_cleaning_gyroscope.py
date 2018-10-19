import pandas as pd

data = pd.read_csv('/Users/bersus96/Desktop/T2.csv', 
                       sep=',')
# 1. Filtering

# 1.1 Filter rows
# convert string to datetime .... Be careful!!! Spelling errors!!!
data['TimeStemp'] = pd.to_datetime(data['TimeStemp'])
# extract date from datetime
data['date'] = [d.date() for d in data['TimeStemp']]
# list the available days
data['date'].unique()
#filter data by date
data_set = data[(data['TimeStemp'] > '2016-04-28 00:00:00') & (data['TimeStemp'] <= '2016-05-05 23:59:59')]

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

data_set_mean = data_set[[column for column in data_set if "Gyroscope" in column and column.endswith("MEAN") ]]

data_set_mean.to_csv('/Users/bersus96/Desktop/T2_cleaned.csv', 
                sep=',', index=False)