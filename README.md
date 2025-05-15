# physicsGR4J model

## Project Introduction

This project combines physical knowledge with deep learning methods to simulate watershed runoff.


## Environment Dependencies

Python 3.8 or higher is recommended.

Install dependencies:
```bash
pip install -r requirements.txt
```

Main dependency packages:
- numpy
- pandas
- matplotlib
- See requirements.txt for others

## Directory Structure

```
.
├── camels/                        # CAMELS dataset related folder
│   ├── basin_list.txt             # List of basin IDs
│   ├── basin_mean_forcing/        # CAMELS meteorological forcing data
│   └── usgs_streamflow/           # USGS measured streamflow data
│
├── libs/                          # Core libraries for hydrological modeling
│   ├── hydrolayer_torch.py        # Implementation of physical/deep learning layers
│   ├── hydrodata.py               # Data loading and processing
│   ├── hydroutils_torch.py        # Other hydrological tools
│   └── tf.py                      # Related tools
│
├── results/                       # Results output folder (automatically archived by station)
│   ├── 14306500/                  # Results for station ID 14306500
│   │   ├── basic_observed_14306500.png      # Overview of observed data
│   │   ├── hybrid_parameter_evolution.png   # Hybrid model parameter evolution plot
│   │   ├── observed_vs_simulated_flow.png   # Comparison of observed and simulated runoff
│   │   ├── predictions.csv                  # Prediction results
│   │   ├── metrics.csv                      # Evaluation metrics
│   │   └── hybrid_parameters.csv            # Parameter evolution data
│   ├── 14306500_common.pth        # Common model weights
│   ├── 14306500_hybrid.pth        # Hybrid model weights
│   └── ...                        # Other stations
│
├── utils/                         # Utilities and configuration
│   ├── config.py                  # Global parameters and configuration file
│   ├── plot_utils.py              # Various plotting functions
│   ├── data_utils.py              # Data processing tools
│   └── evaluation_utils.py        # Evaluation metrics tools
│
├── main.py                  # Main program entry point (main flow for training, evaluation, visualization, etc.)
├── requirements.txt               # List of Python dependency packages
└── README.md                      # Project documentation
```

**Notes:**
- `camels/`: Contains the original dataset and related instructions. It is recommended not to change the structure arbitrarily.
- `data/`: Can be used to store custom or supplementary data.
- `libs/`: Core hydrological modeling and data processing code, easy to extend and maintain.
- `results/`: Each station has a subfolder, and all related images, results, and model weights are automatically archived.
- `utils/`: Stores all utility functions and configuration files, easy to maintain and reuse.
- `main.py`: Main entry point, including parameter parsing, training, evaluation, visualization, and other main processes.

## Data and Code Availability

The datasets for Hybrid-GR4J is openly available at https://gdex.ucar.edu/dataset/camels.html and the code used can be accessed from github: https://github.com/mbSun1/Hybrid_GR4J_PhisicalDL


## Quick Start

1. **Prepare Data**
   Place the original hydrological data into the `camels/` folder. The format must be consistent with the code requirements.

2. **Configure Parameters**
   Modify the parameter configurations in `utils/config.py` as needed.

3. **Run the Main Program**
   ```bash
   python main.py
   ```

4. **View Results**
   Result images and outputs will be automatically saved in the `results/station_ID/` folder.



