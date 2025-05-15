import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, Union, List
import logging

logger = logging.getLogger(__name__)

class DataforIndividual:
    """Individual basin data loading class
    
    Used to load and process meteorological and flow data for a single basin.
    
    Attributes:
        working_path (str): Working directory path
        basin_id (str): Basin ID
    """
    
    def __init__(self, working_path: str, basin_id: str):
        """Initialize data loader
        
        Args:
            working_path: Working directory path
            basin_id: Basin ID
        """
        self.working_path = working_path
        self.basin_id = basin_id

    def check_validation(self, basin_list: pd.DataFrame, basin_id: str) -> None:
        """Validate basin ID
        
        Args:
            basin_list: DataFrame of basin list
            basin_id: Basin ID to validate
            
        Raises:
            AssertionError: If basin ID is invalid
        """
        try:
            assert isinstance(basin_id, str), "Basin ID must be a string"
            assert (len(basin_id) == 8 and basin_id.isdigit()), "Basin ID must be 8 digits"
            assert (basin_id in basin_list.values), "Basin ID must be in basin_list.txt"
        except AssertionError as e:
            logger.error(f"Basin ID validation failed: {str(e)}")
            raise

    def load_forcing_data(self, working_path: str, huc_id: str, basin_id: str) -> Tuple[pd.DataFrame, int]:
        """Load meteorological forcing data
        
        Args:
            working_path: Working directory path
            huc_id: HUC ID
            basin_id: Basin ID
            
        Returns:
            Tuple[pd.DataFrame, int]: Meteorological data DataFrame and basin area
            
        Raises:
            FileNotFoundError: If data file does not exist
        """
        try:
            forcing_path = os.path.join(working_path, 'camels', 'basin_mean_forcing', 'daymet', huc_id,
                                      basin_id + '_lump_cida_forcing_leap.txt')
            
            if not os.path.exists(forcing_path):
                raise FileNotFoundError(f"Forcing data file not found: {forcing_path}")
            
            forcing_data = pd.read_csv(forcing_path, sep="\s+|;|:", header=0, skiprows=3, engine='python')
            forcing_data.rename(columns={"Mnth": "Month"}, inplace=True)
            forcing_data['date'] = pd.to_datetime(forcing_data[['Year', 'Month', 'Day']])
            forcing_data['dayl(day)'] = forcing_data['dayl(s)'] / 86400
            forcing_data['tmean(C)'] = (forcing_data['tmin(C)'] + forcing_data['tmax(C)']) / 2

            # Read area
            with open(forcing_path, 'r') as fp:
                content = fp.readlines()
                area = int(content[2])

            return forcing_data, area
        except Exception as e:
            logger.error(f"Failed to load forcing data: {str(e)}")
            raise

    def load_flow_data(self, working_path: str, huc_id: str, basin_id: str, area: int) -> pd.DataFrame:
        """Load flow data
        
        Args:
            working_path: Working directory path
            huc_id: HUC ID
            basin_id: Basin ID
            area: Basin area
            
        Returns:
            pd.DataFrame: Flow data DataFrame
            
        Raises:
            FileNotFoundError: If data file does not exist
        """
        try:
            flow_path = os.path.join(working_path, 'camels', 'usgs_streamflow', huc_id,
                                   basin_id + '_streamflow_qc.txt')
            
            if not os.path.exists(flow_path):
                raise FileNotFoundError(f"Flow data file not found: {flow_path}")
            
            flow_data = pd.read_csv(flow_path, sep="\s+", names=['Id', 'Year', 'Month', 'Day', 'Q', 'QC'],
                                  header=None, engine='python')
            flow_data['date'] = pd.to_datetime(flow_data[['Year', 'Month', 'Day']])
            flow_data['flow(mm)'] = 28316846.592 * flow_data['Q'] * 86400 / (area * 10 ** 6)  # Unit conversion

            return flow_data
        except Exception as e:
            logger.error(f"Failed to load flow data: {str(e)}")
            raise

    def load_data(self) -> pd.DataFrame:
        """Load basin data
        
        Returns:
            pd.DataFrame: Merged meteorological and flow data
            
        Raises:
            FileNotFoundError: If data file does not exist
            ValueError: If data format is incorrect
        """
        try:
            # Load basin list
            basin_list_path = os.path.join(self.working_path, 'camels', 'basin_list.txt')
            if not os.path.exists(basin_list_path):
                raise FileNotFoundError(f"Basin list file not found: {basin_list_path}")
            
            basin_list = pd.read_csv(basin_list_path, sep='\t', header=0, 
                                   dtype={'HUC': str, 'BASIN_ID': str})
            
            # Validate basin ID
            self.check_validation(basin_list, self.basin_id)
            
            # Get HUC ID
            huc_id = basin_list[basin_list['BASIN_ID'] == self.basin_id]['HUC'].values[0]
            
            # Load data
            forcing_data, area = self.load_forcing_data(self.working_path, huc_id, self.basin_id)
            flow_data = self.load_flow_data(self.working_path, huc_id, self.basin_id, area)
            
            # Merge data
            merged_data = pd.merge(forcing_data, flow_data, on='date')
            
            # Filter date range
            merged_data = merged_data[(merged_data['date'] >= datetime(1980, 10, 1)) &
                                    (merged_data['date'] <= datetime(2019, 9, 30))]
            
            # Set index and select required columns
            merged_data = merged_data.set_index('date')
            pd_data = merged_data[['prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)', 'flow(mm)']]
            
            logger.info(f'Successfully loaded data for basin #{self.basin_id} in HUC #{huc_id}')
            return pd_data
        except Exception as e:
            logger.error(f"Failed to load basin data: {str(e)}")
            raise
