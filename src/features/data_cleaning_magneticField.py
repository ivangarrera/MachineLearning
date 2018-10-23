from basic_data_cleaning import BasicCleaning


data_set = BasicCleaning.CleanData(path_to_data=r'D:\Ing. Informatica\Cuarto\Machine Learning\T2.csv')

data_set = data_set[[column for column in data_set if \
                          "MagneticField" in column and column.endswith("MEAN")]]

data_set.to_csv('D:\Ing. Informatica\Cuarto\Machine Learning\T2_cleaned_magneticField.csv', 
                sep=',', index=False)

