from basic_data_cleaning import BasicCleaning


data_set = BasicCleaning.CleanData(path_to_data=r'C:\Users\ivangarrera\Desktop\T2.csv')

data_set = data_set[[column for column in data_set if \
                          "Gyroscope" in column and column.endswith("MEAN") ]]

data_set.to_csv(r'C:\Users\ivangarrera\Desktop\T2_cleaned.csv',
                sep=',', index=False)