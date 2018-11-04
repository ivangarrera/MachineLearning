# -*- coding: utf-8 -*-

import pandas as pd


class BasicCleaning:
    @classmethod
    def CleanData(cls, path_to_data, var):
        data_set = pd.read_csv(path_to_data, sep=',')
        
        # Convert string to datetime
        data_set['TimeStemp'] = pd.to_datetime(data_set['TimeStemp'])
        
        # filter data by date
        data_set = data_set[(data_set['TimeStemp'] > '2016-04-30 00:00:00') & \
                            (data_set['TimeStemp'] <= '2016-05-01 23:59:59')]
        
        data_set = data_set [[column for column in data_set if var in column and column.endswith("MEAN")]]


        # Remove strings columns
       # for column in data_set:
       #     if type(data_set[column][0]) is str:
       #         data_set = data_set.drop(column, axis=1)
        
        # Remove columns with all null values
        for column in data_set:
            if data_set[column].isnull().all():
                data_set = data_set.drop(column, axis=1)
        
        # Remove duplicate rows
        data_set = data_set.drop_duplicates()
        
        # Remove rows with any null value
        data_set = data_set.dropna()
        
        # Remove version, because 2.3.3 is not a valid float value and PCA could fail
       # data_set = data_set.drop('Version', axis=1)
        
        return data_set