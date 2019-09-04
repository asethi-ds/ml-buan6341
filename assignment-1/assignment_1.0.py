import pandas as pd 
import numpy as np
import datetime as dt
import time
import os
import glob
from configparser import SafeConfigParser, ConfigParser

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
            energy_source_data = pd.concat((pd.read_csv(f) for f in all_source_files))
    return energy_source_data


def feature_engineering(energy_souce_data):
    energy_source_data['date']        = pd.to_datetime(energy_source_data['date'])
    energy_source_data["month"]       = energy_source_data["date"].dt.month
    energy_source_data["week_num"]    = energy_source_data["date"].dt.week
    # Marking the flag if the launch day falls on the weekend
    energy_source_data["is_weekend"]  = energy_source_data["date"].dt.weekday.apply(lambda x: 1 if x > 5 else 0)
    return energy_source_data



config_file_name='loc_config.ini'

energy_source_data=import_source_files(config_file_name)
print(energy_source_data.shape)
energy_source_data_features=feature_engineering(energy_source_data)

print(energy_source_data_features.shape)

