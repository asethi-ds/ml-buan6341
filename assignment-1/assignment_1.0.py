# Assignment 1
# Author : asethi

import pandas as pd 
import numpy as np
import datetime as dt
import time
import os
import glob
from configparser import SafeConfigParser, ConfigParser
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from statsmodels.tools.tools import add_constant

def import_source_files(config_file_name):
    #passing config file information for source file path
    parser = ConfigParser()
    parser.read(config_file_name)
    #Getting source directory path and importing files
    source_dir_path= parser.get('paths', 'source_dir')
    source_dir = os.listdir(source_dir_path)
    if(len(source_dir)):
        all_source_files = glob.glob(source_dir_path + "/*.csv")
        for filename in all_source_files:
        #converting csv into dataframe
            data = pd.concat((pd.read_csv(f) for f in all_source_files))
    return data


def feature_engineering(data):
    data['date']        = pd.to_datetime(data['date'])
    data["month"]       = data["date"].dt.month
    data["week_num"]    = data["date"].dt.week
    # Marking the flag if the launch day falls on the weekend
    data['weekday']     = data['date'].dt.weekday
    data["is_weekend"]  = data["date"].dt.weekday.apply(lambda x: 1 if x > 5 else 0)
    data["hour_of_day"] = data["date"].dt.hour
    data['time_of_day'] = data["hour_of_day"].apply(assign_time_of_day)
    
    feature_categorical_ohe = ['month','is_weekend','time_of_day','weekday']   
    
    data = pd.get_dummies(data,columns = feature_categorical_ohe)
    
    #data  = pd.get_dummies(data,columns=data[['month']])
    #print(data.dtypes)
    return data


def assign_time_of_day(hour_val):
    if hour_val < 6:
        time = 'sleep_time'
    elif hour_val < 9:
        time = 'morning_time'
    elif hour_val <18:
        time = 'work_hours'
    else:
        time = 'night_time'
    return time


def calculate_vif(X, thresh):
    
    feature_ohe_out = ['month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'is_weekend_0','is_weekend_1',
                'time_of_day_morning_time', 'time_of_day_night_time','time_of_day_sleep_time', 'time_of_day_work_hours', 'weekday_0',
                    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5','weekday_6']

    #for var in feature_ohe_out:
     #   X[var]=X[var].astype(float)
    #print((X.dtypes))
    
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix) for ix in range(X.iloc[:, variables].shape[1])]
          
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            del variables[maxloc]
            dropped = True
    return X.iloc[:, variables]

    
config_file_name='loc_config.ini'
independent_original=['T1', 'RH_1', 'T2', 'RH_2', 'T3','RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
                        'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed','Visibility', 'Tdewpoint', 'rv1', 'rv2']

independent_updated=['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3','RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6',
    'RH_6', 'T7', 'RH_7', 'T8','RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed','Visibility', 'Tdewpoint',
    'rv1', 'rv2', 'week_num', 'hour_of_day','month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'is_weekend_0','is_weekend_1',
    'time_of_day_morning_time', 'time_of_day_night_time','time_of_day_sleep_time', 'time_of_day_work_hours', 'weekday_0',
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5','weekday_6']

energy_source_data=import_source_files(config_file_name)
#print(energy_source_data.shape)
energy_source_data_features=feature_engineering(energy_source_data)
#print(energy_source_data_features.columns)

#print(energy_source_data_features.dtypes)
energy_source_data_workset=calculate_vif(energy_source_data_features[independent_updated],10.0)

