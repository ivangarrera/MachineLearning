import pandas as pd

data_set = pd.read_csv('/Users/bersus96/Desktop/T2.csv', 
                       sep=',')
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

data_set_mean = data_set[[column for column in data_set if column.endswith("MEAN")]]

data_set.to_csv('/Users/bersus96/Desktop/T2_cleaned.csv', 
                sep=',', index=False)