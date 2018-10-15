import pandas as pd

data_set = pd.read_csv('D:\Ing. Informatica\Cuarto\Machine Learning\T2.csv',
                       sep=',')
df = data_set
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
      
data_set.to_csv('D:\Ing. Informatica\Cuarto\Machine Learning\T2_cleaned.csv', 
                sep=',', index=False)