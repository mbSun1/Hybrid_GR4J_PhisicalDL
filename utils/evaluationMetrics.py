import numpy as np
import math

def calculate_nse(observed, simulated):
    """Calculate Nash-Sutcliffe Efficiency (NSE)"""
    obs_mean = np.mean(observed)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - obs_mean) ** 2)
    nse = 1 - numerator / denominator
    return round(nse, 2)



def calculate_mse(observed, simulated):
    """Calculate Mean Squared Error (MSE)"""
    mse = np.mean((observed - simulated) ** 2)
    return round(mse, 2)



def calculate_rmse(observed, simulated):
    """Calculate Root Mean Square Error (RMSE)"""
    mse = np.mean((observed - simulated) ** 2)
    rmse = np.sqrt(mse)
    return round(rmse, 2)



def calculate_nrmse(observed, simulated):
    """Calculate Normalized Root Mean Square Error (NRMSE)"""
    rmse = np.sqrt(np.mean((observed - simulated) ** 2))
    obs_std = np.std(observed)
    nrmse = rmse / obs_std
    return round(nrmse, 2)


def calculate_r(observed, simulated):
    """Calculate Pearson correlation coefficient (R)"""
    r = np.corrcoef(observed, simulated)[0, 1]
    return round(r, 2)



def calculate_pbias(observed, simulated):
    """Calculate Percent Bias (PBIAS)"""
    pbias = ((np.sum(simulated - observed)) / np.sum(observed)) * 100
    return round(pbias, 2)



def calculate_kge(observed, simulated):
    """Calculate Kling-Gupta Efficiency (KGE)"""
    r = np.corrcoef(observed, simulated)[0, 1]
    beta = np.std(simulated) / np.std(observed)
    gamma = (np.std(simulated) / np.mean(simulated)) / (np.std(observed) / np.mean(observed))

    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return round(kge, 2)



def calculate_mae(observed, simulated):
    """Calculate Mean Absolute Error (MAE)"""
    mae = np.mean(np.abs(observed - simulated))
    return round(mae, 2)



def calculate_mape(observed, simulated):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    mape = 100 * np.mean(np.abs((observed - simulated) / observed))
    return round(mape, 2)



def calculate_tpe(observed, simulated):
    """Calculate Peak Flow Error (TPE) for top 2% of observed flow sequences"""
    observed_sorted = sorted(observed, reverse=True)
    m = math.ceil(len(observed) * 0.05)
    top_obs_index = sorted(range(len(observed)), key=lambda i: observed[i], reverse=True)[:m]
    top_sim_value = [simulated[i] for i in top_obs_index]
    tpe = sum(abs(top_sim_value[i] - observed_sorted[i]) for i in range(m)) / sum(observed_sorted[:m])
    return round(tpe, 2)
