"""
Configuration file, contains all configurable parameters
"""

# Model configuration
MODEL_CONFIG = {
    'hybrid': True,    # Whether to train the hybrid model
    'physical': False,   # Whether to train the physical model
    'common': True,    # Whether to train the common model
}

# Dataset configuration
DATASET_CONFIG = {
    'basin_id': '14306500',  # Basin ID
    'training_start': '1980-01-01',  # Training set start time
    'training_end': '2000-12-31',    # Training set end time
    'testing_start': '2001-01-01',   # Testing set start time
    'testing_end': '2014-12-31',     # Testing set end time
    'wrap_length': 2190,  # Data wrapping length, recommended to be no less than 5 years (1825 days)
}

# Model training configuration
TRAIN_CONFIG = {
    'seed': 42,            # Fixed random seed
    'num_filters': 16,     # Increase the number of CNN layer filters
    'ep_number': 200,      # Increase the maximum number of training epochs
    'lrate': 0.005,        # Decrease initial learning rate
    'batch_size': 10000,    # Decrease batch size
}

# Plotting configuration
PLOT_CONFIG = {
    'colors': {
        'hybrid': '#e41a1c',    # Hybrid model color
        'physical': '#377eb8',  # Physical model color
        'common': '#4daf4a'     # Common model color
    },
    'titles': {
        'hybrid': 'Hybrid DL model',
        'physical': 'Physical NN model',
        'common': 'Common NN model'
    },
    'data_overview_figsize': (15, 15),  # Data overview figure size, adjusted to fit 6 subplots
    'result_figsize': (15, 4),          # Result figure size
    'ylim': [0, 30]                     # y-axis range
} 