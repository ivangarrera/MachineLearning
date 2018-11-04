from basic_data_cleaning import BasicCleaning


#data_set = BasicCleaning.CleanData(path_to_data=r'C:\Users\ivangarrera\Desktop\T2.csv')
data_set = BasicCleaning.CleanData(path_to_data=r'D:\Ing. Informatica\Cuarto\Machine Learning\T2.csv',var="Gyroscope")



#data_set.to_csv(r'C:\Users\ivangarrera\Desktop\T2_cleaned.csv',sep=',', index=False)
data_set.to_csv(r'D:\Ing. Informatica\Cuarto\Machine Learning\T2_cleaned_gyroscope.csv',sep=',', index=False)
