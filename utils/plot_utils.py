import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.config import PLOT_CONFIG
import matplotlib as mpl
import os

# Set Chinese and English fonts
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to display Chinese labels normally
plt.rcParams['axes.unicode_minus'] = False  # Used to display the minus sign normally
plt.rcParams['font.family'] = 'sans-serif'  # Set font family
plt.rcParams['mathtext.fontset'] = 'stix'  # Set math font

def plot_data_overview(hydrodata, basin_id):
    """Plot data overview"""
    fig, [ax1, ax2, ax3, ax4, ax5, ax6] = plt.subplots(nrows=6, ncols=1, sharex='row', figsize=PLOT_CONFIG['data_overview_figsize'])
    
    # Plot each variable
    ax1.plot(hydrodata.index, hydrodata['prcp(mm/day)'], color='#377eb8')
    ax2.plot(hydrodata.index, hydrodata['tmean(C)'], color='#e41a1c')
    ax3.plot(hydrodata.index, hydrodata['dayl(day)'], color='#ff7f00')
    ax4.plot(hydrodata.index, hydrodata['srad(W/m2)'], color='#984ea3')
    ax5.plot(hydrodata.index, hydrodata['vp(Pa)'], color='#a65628')
    ax6.plot(hydrodata.index, hydrodata['flow(mm)'], color='#4daf4a')
    
    # Set titles and labels
    ax1.set_title(f"Basin {basin_id}")
    ax1.set_ylabel("prcp(mm/day)")   # Precipitation
    ax2.set_ylabel("tmean(C)")       # Mean temperature
    ax3.set_ylabel("dayl(day)")      # Daylength
    ax4.set_ylabel("srad(W/m2)")     # Solar radiation
    ax5.set_ylabel("vp(Pa)")         # Vapor pressure
    ax6.set_ylabel("flow(mm)")       # Flow
    ax6.set_xlabel("Date")
    
    plt.tight_layout()
    os.makedirs(f'results/{basin_id}', exist_ok=True)
    plt.savefig(f'results/{basin_id}/basic_observed_{basin_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluation_plot(ax, plot_set, flow_col, color, nse, basin_id):
    """Plot evaluation results for a single model"""
    ax.plot(plot_set.index, plot_set['flow_obs'], color='k', label='Observed')
    ax.plot(plot_set.index, plot_set[flow_col], color=color, label='Simulated')
    ax.set_ylabel('Flow (mm/day)')
    ax.set_ylim(PLOT_CONFIG['ylim'])
    ax.legend()
    ax.annotate(f'NSE = {nse:.3f}', xy=(0.05, 0.8), xycoords='axes fraction', size=12)

def plot_model_results(evaluate_set, model_predictions, model_nse, basin_id, testing_start, testing_end):
    """Plot results for all models"""
    plot_set = evaluate_set[evaluate_set.index.isin(pd.date_range(testing_start, testing_end))]
    n_models = len(model_predictions)

    # Ensure axes is always an iterable object
    if n_models == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=PLOT_CONFIG['result_figsize'])
        axes = [axes]  # Convert single axes object to list
    else:
        fig, axes = plt.subplots(nrows=n_models, ncols=1, sharex='row', 
                                figsize=(PLOT_CONFIG['result_figsize'][0], 
                                        PLOT_CONFIG['result_figsize'][1] * n_models))

    for i, (model_name, ax) in enumerate(zip(model_predictions.keys(), axes)):
        evaluation_plot(ax, plot_set, f'flow_{model_name}', 
                       PLOT_CONFIG['colors'][model_name], 
                       model_nse[model_name], basin_id)
        ax.annotate(PLOT_CONFIG['titles'][model_name], 
                   xy=(0.05, 0.9), xycoords='axes fraction', size=12)

    plt.tight_layout()
    # plt.show()  # Commented out to avoid clearing the image content when saving

def plot_parameter_evolution(param_history, save_path=None):
    """Plot hydrological parameter evolution process"""
    param_names = {
        'x1': 'Production store capacity (mm)',
        'x2': 'Groundwater exchange coefficient',
        'x3': 'Routing store capacity (mm)'
    }
    
    epochs = range(1, len(param_history['x1']) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    axes = axes.ravel()
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    
    for (param_key, param_values), ax, color in zip(param_history.items(), axes, colors): # Changed param_name to param_key to avoid conflict
        ax.plot(epochs, param_values, color=color, linewidth=2)
        ax.set_xlabel('Training Epochs')
        ax.set_ylabel(param_names[param_key])
        ax.set_title(f'Evolution of {param_names[param_key]}')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    # else:
    #     plt.show()  # Commented out to maintain consistency with previous changes