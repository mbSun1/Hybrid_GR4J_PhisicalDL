import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from typing import Tuple, Dict, Optional, Union, List

# Ignore specific convolution layer warnings
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created")

# Import libraries developed for this study
from libs.hydrolayer_torch import PRNNLayer, ConvLayer, ScaleLayer
from libs.hydrodata import DataforIndividual
from libs import hydroutils_torch as hydroutils

# Import utility modules
from utils.plot_utils import plot_data_overview, plot_model_results, plot_parameter_evolution
from utils.data_utils import generate_train_test, normalize, addYears
from utils.evaluation_utils import calc_nse
from utils.config import MODEL_CONFIG, DATASET_CONFIG, TRAIN_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------------------
# 1. Environment Setup

def setup_environment() -> str:
    """Set up the runtime environment
    
    Returns:
        str: Working directory path
    """
    try:
        # Set random seed
        np.random.seed(TRAIN_CONFIG['seed'])
        torch.manual_seed(TRAIN_CONFIG['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(TRAIN_CONFIG['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(TRAIN_CONFIG['seed'])
        
        # Get current working directory
        working_path = os.path.dirname(os.path.abspath(__file__))
        
        # Ensure required directories exist
        required_dirs = [
            os.path.join(working_path, 'results'),
            os.path.join(working_path, 'camels'),
            os.path.join(working_path, 'camels', 'basin_mean_forcing'),
            os.path.join(working_path, 'camels', 'basin_mean_forcing', 'daymet'),
            os.path.join(working_path, 'camels', 'usgs_streamflow')
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
        
        # Change to working directory
        os.chdir(working_path)
        
        return working_path
    except Exception as e:
        logger.error(f"Environment setup failed: {str(e)}")
        raise

# ------------------------------------------------------------------------------------------------------------
# 2. Data Preparation

def prepare_data(working_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame, str]:
    """Prepare training and testing data
    
    Args:
        working_path: Working directory path
        device: Computing device
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame, str]: 
            train input, train output, test input, test output, test set, basin ID
    """
    try:
        # Load data
        basin_id = DATASET_CONFIG['basin_id']
        
        # Ensure basin_id is 8 digits
        if not basin_id.isdigit() or len(basin_id) != 8:
            # If numeric but not 8 digits, pad with zeros
            if basin_id.isdigit():
                basin_id = basin_id.zfill(8)
            else:
                raise ValueError("Basin ID must be 8 digits")
        
        # Create station-specific results directory
        station_results_dir = os.path.join(working_path, 'results', basin_id)
        os.makedirs(station_results_dir, exist_ok=True)
        logger.info(f"Created station results directory: {station_results_dir}")
        
        hydrodata = DataforIndividual(working_path, basin_id).load_data()
        
        # Plot data overview
        plot_data_overview(hydrodata, basin_id)
        
        # Split dataset
        train_set = hydrodata[hydrodata.index.isin(pd.date_range(
            DATASET_CONFIG['training_start'], 
            DATASET_CONFIG['training_end']
        ))]
        test_set = hydrodata[hydrodata.index.isin(pd.date_range(
            DATASET_CONFIG['testing_start'], 
            DATASET_CONFIG['testing_end']
        ))]
        
        logger.info(f"Training set from {DATASET_CONFIG['training_start']} to {DATASET_CONFIG['training_end']}, shape: {train_set.shape}")
        logger.info(f"Testing set from {DATASET_CONFIG['testing_start']} to {DATASET_CONFIG['testing_end']}, shape: {test_set.shape}")
        
        # Generate training and testing samples
        train_x, train_y, test_x, test_y = generate_train_test(
            train_set, test_set, wrap_length=DATASET_CONFIG['wrap_length']
        )
        
        # Convert to PyTorch tensors and move to device
        train_x = torch.FloatTensor(train_x).to(device)
        train_y = torch.FloatTensor(train_y).to(device)
        test_x = torch.FloatTensor(test_x).to(device)
        test_y = torch.FloatTensor(test_y).to(device)
        
        return train_x, train_y, test_x, test_y, test_set, basin_id
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise

# ------------------------------------------------------------------------------------------------------------
# 3. Model Definition
class HybridModel(nn.Module):
    """Hybrid hydrological model combining physical and deep learning models"""
    
    def __init__(self, input_shape: Union[torch.Size, Tuple[int, ...]], 
                 num_filters: int, 
                 model_type: str = 'hybrid', 
                 device: Optional[torch.device] = None):
        """Initialize hybrid model
        
        Args:
            input_shape: Shape of input tensor
            num_filters: Number of filters in convolutional layers
            model_type: Model type, options: 'hybrid', 'physical', 'common'
            device: Computing device
        """
        super(HybridModel, self).__init__()
        self.model_type = model_type
        self.device = device
        
        # Ensure input_shape is a tuple
        if isinstance(input_shape, torch.Size):
            input_shape = tuple(input_shape)
        
        # Get number of input features
        if len(input_shape) == 3:
            input_channels = input_shape[2]
        else:
            input_channels = input_shape[1]  # If input is 2D tensor
            
        if model_type == 'hybrid':
            self.hydro = PRNNLayer().to(device)
            self.scale = ScaleLayer().to(device)
            self.conv1 = ConvLayer(input_channels + 1, num_filters, kernel_size=10, padding='causal').to(device)
            self.conv2 = ConvLayer(num_filters, 1, kernel_size=1, padding='causal').to(device)
            
        elif model_type == 'physical':
            self.hydro = PRNNLayer().to(device)
            
        elif model_type == 'common':
            self.conv1 = ConvLayer(input_channels, num_filters, kernel_size=10, padding='causal').to(device)
            self.conv2 = ConvLayer(num_filters, 1, kernel_size=1, padding='causal').to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        if self.model_type == 'hybrid':
            hydro_out = self.hydro(x)
            x_hydro = torch.cat([x, hydro_out], dim=-1)
            x_scale = self.scale(x_hydro)
            x_scale = x_scale.permute(0, 2, 1)  # Adjust dimension order for Conv1d
            cnn_out = self.conv1(x_scale)
            cnn_out = self.conv2(cnn_out)
            return cnn_out.permute(0, 2, 1)  # Restore original dimension order
            
        elif self.model_type == 'physical':
            return self.hydro(x)
            
        elif self.model_type == 'common':
            x = x.permute(0, 2, 1)
            cnn_out = self.conv1(x)
            cnn_out = self.conv2(cnn_out)
            return cnn_out.permute(0, 2, 1)

# ------------------------------------------------------------------------------------------------------------
# 4. Model Training and Testing

def train_model(model, train_x, train_y, ep_number, lrate, save_path, device):
    """Train PyTorch model"""
    # Use pathlib for path handling
    save_path = Path(save_path)
    try:
        # Ensure parent directory of save path exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory: {e}")
        # If directory creation fails, use current directory
        save_path = os.path.join(os.getcwd(), save_path.name)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lrate)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, 
        verbose=True, threshold=0.001, cooldown=5, min_lr=lrate/100
    )
    
    # Define early stopping
    best_nse = float('-inf')
    patience = 30
    patience_counter = 0
    
    # Record hydrological parameter history
    param_history = {
        'x1': [],  # Maximum storage capacity
        'x2': [],  # Soil moisture index
        'x3': []   # Groundwater recharge coefficient
    }
    
    print("\nStart training...")
    print(f"{'Epoch':^6} {'Loss':^12} {'NSE':^12} {'LR':^12} {'x1':^8} {'x2':^8} {'x3':^8}")
    print("-" * 65)
    
    for epoch in range(ep_number):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(train_x)
        
        # Compute loss
        loss = hydroutils.nse_loss(train_y, outputs)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute NSE
        with torch.no_grad():
            nse = hydroutils.nse_metrics(train_y, outputs)
        
        # Update learning rate
        scheduler.step(nse)
        
        # Record hydrological parameters
        if hasattr(model, 'hydro'):
            x1_val = float(model.hydro.x1 * 2000)
            x2_val = float(model.hydro.x2 * 40 - 20)
            x3_val = float(model.hydro.x3 * 300)
            param_history['x1'].append(x1_val)
            param_history['x2'].append(x2_val)
            param_history['x3'].append(x3_val)
        
        # Print training info
        current_lr = optimizer.param_groups[0]['lr']
        if hasattr(model, 'hydro'):
            print(f"{epoch+1:^6d} {loss.item():^12.4f} {nse.item():^12.4f} {current_lr:^12.6f} {x1_val:^8.2f} {x2_val:^8.2f} {x3_val:^8.2f}")
        else:
            print(f"{epoch+1:^6d} {loss.item():^12.4f} {nse.item():^12.4f} {current_lr:^12.6f}")
        
        # Early stopping check
        if nse > best_nse:
            best_nse = nse
            patience_counter = 0
            try:
                # Try to save model, but do not print save path
                torch.save(model.state_dict(), str(save_path))
            except Exception as e:
                print(f"\nFailed to save model: {e}")
                # If saving fails, try using current directory
                alt_save_path = os.path.join(os.getcwd(), f"model_{epoch+1}.pth")
                try:
                    torch.save(model.state_dict(), str(alt_save_path))
                except Exception as e2:
                    print(f"Backup path save also failed: {e2}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! Best NSE: {best_nse:.4f}")
                break
    
    print("\nTraining complete!")
    print(f"Final NSE: {best_nse:.4f}")
    return param_history

def test_model(model, test_x, save_path, device):
    """Test PyTorch model"""
    try:
        # Load model weights
        state_dict = torch.load(save_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print(f"Tried loading from: {save_path}")
        raise
    
    model.eval()
    with torch.no_grad():
        pred_y = model(test_x)
    
    return pred_y.cpu().numpy()

def train_and_test_models(train_x, train_y, test_x, test_y, basin_id, working_path, test_set):
    """Train and test all configured models"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Use pathlib for path handling
    working_path = Path(working_path)
    station_results_dir = os.path.join(working_path, 'results', basin_id)
    
    # Check if results directory exists
    if not os.path.exists(station_results_dir):
        try:
            os.makedirs(station_results_dir, exist_ok=True)
            print(f"Created station results directory: {station_results_dir}")
        except Exception as e:
            print(f"Failed to create station results directory: {e}")
            # If creation fails, use current directory
            station_results_dir = os.path.join(os.getcwd(), 'results', basin_id)
            if not os.path.exists(station_results_dir):
                os.makedirs(station_results_dir, exist_ok=True)
            print(f"Will use results/{basin_id} folder in current directory: {station_results_dir}")
    
    model_predictions = {}
    param_histories = {}
    
    # Train and test hybrid model
    if MODEL_CONFIG['hybrid']:
        print("\nTraining and testing hybrid model...")
        save_path = os.path.join(station_results_dir, f'{basin_id}_hybrid.pth')
        model = HybridModel(train_x.shape, 
                          num_filters=TRAIN_CONFIG['num_filters'], 
                          model_type='hybrid',
                          device=device)
        param_history = train_model(model, train_x, train_y, 
                                  ep_number=TRAIN_CONFIG['ep_number'], 
                                  lrate=TRAIN_CONFIG['lrate'], 
                                  save_path=str(save_path),
                                  device=device)
        param_histories['hybrid'] = param_history
        
        model = HybridModel(test_x.shape, 
                          num_filters=TRAIN_CONFIG['num_filters'], 
                          model_type='hybrid',
                          device=device)
        flow_hybrid = test_model(model, test_x, str(save_path), device)
        model_predictions['hybrid'] = flow_hybrid

    # Train and test physical model
    if MODEL_CONFIG['physical']:
        print("\nTraining and testing physical model...")
        save_path = os.path.join(station_results_dir, f'{basin_id}_physical.pth')
        model = HybridModel(train_x.shape, 
                          num_filters=TRAIN_CONFIG['num_filters'], 
                          model_type='physical',
                          device=device)
        param_history = train_model(model, train_x, train_y, 
                                  ep_number=TRAIN_CONFIG['ep_number'], 
                                  lrate=TRAIN_CONFIG['lrate'], 
                                  save_path=str(save_path),
                                  device=device)
        param_histories['physical'] = param_history
        
        model = HybridModel(test_x.shape, 
                          num_filters=TRAIN_CONFIG['num_filters'], 
                          model_type='physical',
                          device=device)
        flow_physical = test_model(model, test_x, str(save_path), device)
        model_predictions['physical'] = flow_physical

    # Train and test common model
    if MODEL_CONFIG['common']:
        print("\nTraining and testing common model...")
        save_path = os.path.join(station_results_dir, f'{basin_id}_common.pth')
        
        # Data normalization
        train_x_nor, train_x_mean, train_x_std = normalize(train_x.cpu().numpy())
        train_y_nor, train_y_mean, train_y_std = normalize(train_y.cpu().numpy())
        
        train_x_nor = torch.FloatTensor(train_x_nor).to(device)
        train_y_nor = torch.FloatTensor(train_y_nor).to(device)
        
        model = HybridModel(train_x_nor.shape, 
                          num_filters=TRAIN_CONFIG['num_filters'], 
                          model_type='common',
                          device=device)
        train_model(model, train_x_nor, train_y_nor, 
                   ep_number=TRAIN_CONFIG['ep_number'], 
                   lrate=TRAIN_CONFIG['lrate'], 
                   save_path=str(save_path),
                   device=device)
        
        model = HybridModel(test_x.shape, 
                          num_filters=TRAIN_CONFIG['num_filters'], 
                          model_type='common',
                          device=device)
        test_x_nor = (test_x.cpu().numpy() - train_x_mean) / train_x_std
        test_x_nor = torch.FloatTensor(test_x_nor).to(device)
        flow_common = test_model(model, test_x_nor, str(save_path), device)
        flow_common = flow_common * train_y_std + train_y_mean
        model_predictions['common'] = flow_common
    
    return model_predictions, param_histories

# ------------------------------------------------------------------------------------------------------------
# 5. Main Function

def main():
    """Main function, batch process all BASIN_IDs"""
    # Set up environment
    working_path = setup_environment()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Read all BASIN_IDs
    basin_list_path = os.path.join(working_path, 'camels', 'basin_list.txt')
    basin_list = pd.read_csv(basin_list_path, sep='\t')
    all_basin_ids = basin_list['BASIN_ID'].astype(str).str.zfill(8).tolist()

    # Create NSE results file
    nse_results_file = os.path.join(working_path, 'results', 'nse_results.txt')
    with open(nse_results_file, 'w') as f:
        f.write("basin_id\thybrid_nse\tphysical_nse\tcommon_nse\n")

    for basin_id in all_basin_ids:
        print(f"\n===== Start processing station {basin_id} =====")
        # Modify global config basin_id
        DATASET_CONFIG['basin_id'] = basin_id
        try:
            # Data preparation
            train_x, train_y, test_x, test_y, test_set, _ = prepare_data(working_path, device)
            # Train and test models
            model_predictions, param_histories = train_and_test_models(
                train_x, train_y, test_x, test_y, basin_id, working_path, test_set
            )
            # Evaluate and visualize results
            model_nse = evaluate_and_visualize(model_predictions, test_set, basin_id, working_path, param_histories)

            # Write NSE values to file
            nse_file = os.path.join(working_path, 'results', 'nse_results.txt')
            with open(nse_file, 'a') as f:
                hybrid_nse = model_nse.get('hybrid', 'NA')
                physical_nse = model_nse.get('physical', 'NA')
                common_nse = model_nse.get('common', 'NA')
                f.write(f"{basin_id}\t{hybrid_nse}\t{physical_nse}\t{common_nse}\n")

            logger.info(f"NSE values saved to: {nse_file}")
        except Exception as e:
            print(f"Station {basin_id} processing failed: {e}")
        print(f"===== Station {basin_id} processing complete =====\n")

def evaluate_and_visualize(model_predictions, test_set, basin_id, working_path, param_histories=None):
    """Evaluate model performance and visualize results"""
    evaluate_set = test_set.loc[:, ['prcp(mm/day)','flow(mm)']]
    evaluate_set['flow_obs'] = evaluate_set['flow(mm)']

    # Add model prediction results
    for model_name, predictions in model_predictions.items():
        evaluate_set[f'flow_{model_name}'] = np.clip(predictions[0, :, :], a_min=0, a_max=None)

    # Calculate evaluation period
    evaluation_start = datetime.strftime(addYears(datetime.strptime(DATASET_CONFIG['testing_start'], '%Y-%m-%d'), 1), '%Y-%m-%d')
    evaluation_end = DATASET_CONFIG['testing_end']

    # Calculate NSE values
    date_range = pd.date_range(evaluation_start, evaluation_end)
    evaluate_set = evaluate_set[evaluate_set.index.isin(date_range)]
    
    model_nse = {}
    for model_name in model_predictions.keys():
        nse_value = calc_nse(evaluate_set['flow_obs'].values, evaluate_set[f'flow_{model_name}'].values)
        model_nse[model_name] = nse_value
        print(f"{model_name} model NSE: {nse_value:.3f}")

    # Create station-specific results directory
    station_results_dir = os.path.join(working_path, 'results', basin_id)
    os.makedirs(station_results_dir, exist_ok=True)
    print(f"Created station results directory: {station_results_dir}")

    # Plot results
    plt.figure(figsize=(15, 10))
    plot_model_results(evaluate_set, model_predictions, model_nse, basin_id, 
                      DATASET_CONFIG['testing_start'], DATASET_CONFIG['testing_end'])
    
    # Save evaluation metrics
    metrics_file = os.path.join(station_results_dir, 'metrics.csv')
    metrics_df = pd.DataFrame({
        'Model': list(model_nse.keys()),
        'NSE': list(model_nse.values())
    })
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Evaluation metrics saved to: {metrics_file}")
    
    # Save prediction results
    predictions_file = os.path.join(station_results_dir, 'predictions.csv')
    evaluate_set.to_csv(predictions_file)
    print(f"Prediction results saved to: {predictions_file}")
    
    # Save visualization results
    plot_file = os.path.join(station_results_dir, 'results.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print(f"Visualization results saved to: {plot_file}")

    # If hydrological parameter history exists, save and plot them
    if param_histories:
        for model_name, param_history in param_histories.items():
            # Save parameter history to CSV
            param_file = os.path.join(station_results_dir, f'{model_name}_parameters.csv')
            pd.DataFrame(param_history).to_csv(param_file, index=False)
            print(f"{model_name} model parameter history saved to: {param_file}")
            
            # Plot parameter evolution
            if model_name in ['hybrid', 'physical'] and all(k in param_history for k in ['x1', 'x2', 'x3']):
                param_plot_file = os.path.join(station_results_dir, f'{model_name}_parameter_evolution.png')
                plot_parameter_evolution(param_history, save_path=param_plot_file)

    return model_nse

if __name__ == "__main__":
    main() 
    main() 