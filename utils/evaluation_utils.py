import numpy as np

def calc_nse(obs, sim):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE)
    Args:
        obs: Array of observed values
        sim: Array of simulated values
    Returns:
        NSE value
    """
    # Ensure inputs are numpy arrays
    obs = np.array(obs)
    sim = np.array(sim)
    
    # Check array lengths
    if len(obs) != len(sim):
        raise ValueError("Observed and simulated values must have the same length")
    
    # Calculate NSE
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    
    # Prevent division by zero
    if denominator == 0:
        return -np.inf
    
    return 1 - numerator / denominator

def calc_kge(obs, sim):
    """
    Calculate Kling-Gupta Efficiency (KGE)
    Args:
        obs: Array of observed values
        sim: Array of simulated values
    Returns:
        KGE value
    """
    # Ensure inputs are numpy arrays
    obs = np.array(obs)
    sim = np.array(sim)
    
    # Check array lengths
    if len(obs) != len(sim):
        raise ValueError("Observed and simulated values must have the same length")
    
    # Calculate statistics
    r = np.corrcoef(obs, sim)[0, 1]  # Correlation coefficient
    alpha = np.std(sim) / np.std(obs)  # Variability ratio
    beta = np.mean(sim) / np.mean(obs)  # Bias ratio
    
    # Calculate KGE
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

def calc_rmse(obs, sim):
    """
    Calculate Root Mean Square Error (RMSE)
    Args:
        obs: Array of observed values
        sim: Array of simulated values
    Returns:
        RMSE value
    """
    # Ensure inputs are numpy arrays
    obs = np.array(obs)
    sim = np.array(sim)
    
    # Check array lengths
    if len(obs) != len(sim):
        raise ValueError("Observed and simulated values must have the same length")
    
    # Calculate RMSE
    return np.sqrt(np.mean((obs - sim) ** 2))

def calc_mae(obs, sim):
    """
    Calculate Mean Absolute Error (MAE)
    Args:
        obs: Array of observed values
        sim: Array of simulated values
    Returns:
        MAE value
    """
    # Ensure inputs are numpy arrays
    obs = np.array(obs)
    sim = np.array(sim)
    
    # Check array lengths
    if len(obs) != len(sim):
        raise ValueError("Observed and simulated values must have the same length")
    
    # Calculate MAE
    return np.mean(np.abs(obs - sim)) 