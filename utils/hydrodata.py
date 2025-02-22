import os
import pandas as pd
from datetime import datetime

class DataforIndividual():
    def __init__(self, working_path, save_data_path, basin_id, time_start, time_end, process_counter):
        self.working_path = working_path
        self.basin_id = basin_id
        self.time_start = time_start
        self.time_end = time_end
        self.save_data_path = save_data_path
        # self.writer = pd.ExcelWriter(self.save_data_filename, engine='openpyxl', mode='a', if_sheet_exists='replace')
        self.process_counter = process_counter

    def check_validation(self, basin_list, basin_id):
        assert isinstance(basin_id, str), "The basin ID should be a string"
        assert (len(basin_id) == 8 and basin_id.isdigit()), "Basin ID can only be represented by 8 digits"
        assert (basin_id in basin_list.values), "Please confirm the basin specified is in basin_list.txt"

    def load_forcing_data(self, working_path, huc_id, basin_id):
        forcing_path = os.path.join(working_path, 'basin_mean_forcing', 'daymet', huc_id,
                                    basin_id + '_lump_cida_forcing_leap.txt')
        forcing_data = pd.read_csv(forcing_path, sep="\s+|;|:", header=0, skiprows=3, engine='python')
        forcing_data.rename(columns={"Mnth": "Month"}, inplace=True)
        forcing_data['date'] = pd.to_datetime(forcing_data[['Year', 'Month', 'Day']])
        forcing_data['dayl(day)'] = forcing_data['dayl(s)'] / 86400
        forcing_data['tmean(C)'] = (forcing_data['tmin(C)'] + forcing_data['tmax(C)']) / 2

        # The third line of the file is the area (unit: square feet)
        with open(forcing_path, 'r') as fp:
            content = fp.readlines()
            area = int(content[2])

        return forcing_data, area

    def load_flow_data(self, working_path, huc_id, basin_id, area):
        flow_path = os.path.join(working_path, 'usgs_streamflow', huc_id,
                                 basin_id + '_streamflow_qc.txt')
        flow_data = pd.read_csv(flow_path, sep="\s+", names=['Id', 'Year', 'Month', 'Day', 'Q', 'QC'],
                                header=None, engine='python')
        flow_data['date'] = pd.to_datetime(flow_data[['Year', 'Month', 'Day']])
        flow_data['flow(mm)'] = 28316846.592/(10**9) * flow_data['Q'] * 86.4 / (area /(10 ** 6))#cubic feet per second to cubic meter per second

        return flow_data

    def load_data(self):
        basin_list = pd.read_csv(os.path.join(self.working_path, 'basin_list.txt'),
                                 sep='\t', header=0, dtype={'HUC': str, 'BASIN_ID': str})
        self.check_validation(basin_list, self.basin_id)
        huc_id = basin_list[basin_list['BASIN_ID'] == self.basin_id]['HUC'].values[0]
        forcing_data, area = self.load_forcing_data(self.working_path, huc_id, self.basin_id)
        flow_data = self.load_flow_data(self.working_path, huc_id, self.basin_id, area)
        flow_data.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)
        merged_data = pd.merge(forcing_data, flow_data, on='date')

        merged_data = merged_data[(merged_data['date'] >= datetime(1980, 1, 1)) &
                                  (merged_data['date'] <= datetime(2010, 1, 1))]

        merged_data = merged_data.set_index('date')
        hydrodata = merged_data[['prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)', 'Month', 'Day', 'flow(mm)']]

        #The data needed is intercepted
        output_data = hydrodata[hydrodata.index.isin(pd.date_range(self.time_start, self.time_end))]

        output_filename = f"{self.basin_id}.csv"
        output_filepath = os.path.join(self.save_data_path, output_filename)
        output_data.to_csv(output_filepath, index=True)
        print('The #{}th data in basin #{} has been successfully processed.'.format(self.process_counter, self.basin_id))
        return output_data



