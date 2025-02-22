from utils.hydrodata import DataforIndividual
import os
import pandas as pd

working_path = './camels' #The folder path where the camels data is indexed
save_data_path = './data' #Set the path to save the processed data
#Set the start time for each site calculation
time_start, time_end = '1980-01-01', '2009-12-31'

# List of basin IDs
basin_list = pd.read_csv(os.path.join(working_path, 'basin_list.txt'),
                        sep='\t', header=0, dtype={'HUC': str, 'BASIN_ID': str})
basin_ids = basin_list['BASIN_ID']

process_counter = 1
for basin_id in basin_ids:
    DataforIndividual(working_path, save_data_path, basin_id, time_start, time_end, process_counter).load_data()
    process_counter += 1
print("All camels data are processed into standard format!")

