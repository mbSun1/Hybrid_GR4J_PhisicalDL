import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_train_test(train_set, test_set, wrap_length):
    """Generate training and testing samples"""
    # Ensure inputs are DataFrames
    if not isinstance(train_set, pd.DataFrame) or not isinstance(test_set, pd.DataFrame):
        raise ValueError("train_set and test_set must be pandas DataFrames")
    
    # Extract data
    train_x_np = train_set.values[:, :-1]
    train_y_np = train_set.values[:, -1:]
    test_x_np = test_set.values[:, :-1]
    test_y_np = test_set.values[:, -1:]

    # Calculate number of training samples
    wrap_number_train = (train_set.shape[0] - wrap_length) // 365 + 1

    # Initialize training data arrays
    train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_y = np.empty(shape=(wrap_number_train, wrap_length, train_y_np.shape[1]))

    # Process test data
    test_x = np.expand_dims(test_x_np, axis=0)
    test_y = np.expand_dims(test_y_np, axis=0)

    # Generate training samples
    for i in range(wrap_number_train):
        train_x[i, :, :] = train_x_np[i * 365:(wrap_length + i * 365), :]
        train_y[i, :, :] = train_y_np[i * 365:(wrap_length + i * 365), :]

    return train_x, train_y, test_x, test_y

def normalize(data):
    """Normalize data"""
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array")
    
    # Calculate mean and standard deviation
    data_mean = np.mean(data, axis=-2, keepdims=True)
    data_std = np.std(data, axis=-2, keepdims=True)
    
    # Prevent division by zero
    data_std = np.maximum(data_std, 1e-6)
    
    # Normalize
    data_scaled = (data - data_mean) / data_std
    
    return data_scaled, data_mean, data_std

def addYears(date_str, years):
    """
    Add a specified number of years to a date string
    Args:
        date_str: Date string or datetime object
        years: Number of years to add
    Returns:
        Date string after adding years
    """
    try:
        if isinstance(date_str, str):
            date = datetime.strptime(date_str, '%Y-%m-%d')
        else:
            date = date_str
        return date.replace(year=date.year + years)
    except Exception as e:
        print(f"Date processing error: {e}")
        return date_str 