from basic_data_cleaning import BasicCleaning

# Read csv and clean data

#data_set = BasicCleaning.CleanData(path_to_data=r'C:\Users\ivangarrera\Desktop\T2.csv',var="Accelerometer")
data_set = BasicCleaning.CleanData(path_to_data=r'D:\Ing. Informatica\Cuarto\Machine Learning\T2.csv',var="Accelerometer")


#data_set.to_csv(r'C:\Users\ivangarrera\Desktop\T2_cleaned.csv',sep=',', index=False)
data_set.to_csv(r'D:\Ing. Informatica\Cuarto\Machine Learning\T2_cleaned_accelerometer.csv',sep=',', index=False)
