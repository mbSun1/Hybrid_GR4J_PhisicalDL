import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Union, List
import logging

logger = logging.getLogger(__name__)

class PRNNLayer(nn.Module):
    """
    Physical-aware RNN Layer
    Uses convolutional layers to implement the routing mechanism.
    """
    def __init__(self):
        """Initialize PRNN layer"""
        super(PRNNLayer, self).__init__()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize trainable parameters
        self.x1 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device))  # ∈ (1/2000, 1)
        self.x2 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device))  # ∈ (0, 1)
        self.x3 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device))  # ∈ (1/300, 1)
        
        # Define routing convolutional layers
        self.uh1_conv = nn.Conv1d(1, 1, kernel_size=30, padding='same', bias=False, device=self.device)
        self.uh2_conv = nn.Conv1d(1, 1, kernel_size=30, padding='same', bias=False, device=self.device)
        
        # Initialize convolution weights
        self._init_conv_weights()
    
    def _init_conv_weights(self) -> None:
        """Initialize convolution weights
        
        Initializes unit hydrograph weights using an exponential decay function and normalizes them.
        """
        try:
            # Initialize unit hydrograph weights
            uh1_weights = torch.zeros(1, 1, 30, device=self.device)
            uh2_weights = torch.zeros(1, 1, 30, device=self.device)
            
            # Use torch.tensor to ensure correct type
            for i in range(30):
                uh1_weights[0, 0, i] = torch.exp(torch.tensor(-i / 5.0, device=self.device))
                uh2_weights[0, 0, i] = torch.exp(torch.tensor(-i / 10.0, device=self.device))
            
            # Normalize weights
            uh1_weights = uh1_weights / uh1_weights.sum()
            uh2_weights = uh2_weights / uh2_weights.sum()
            
            # Set convolution layer weights
            self.uh1_conv.weight.data = uh1_weights
            self.uh2_conv.weight.data = uh2_weights
        except Exception as e:
            logger.error(f"Failed to initialize convolution weights: {str(e)}")
            raise
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            inputs: Input tensor, shape [batch_size, time_steps, features]
                   features include precipitation, temperature, and daylength
            
        Returns:
            torch.Tensor: Simulated flow, shape [batch_size, time_steps, 1]
        """
        try:
            # Ensure input is float32 type
            inputs = inputs.to(torch.float32)
            
            p = inputs[:, :, 0]
            t = inputs[:, :, 1]
            dayl = inputs[:, :, 2]
            
            # Calculate potential evapotranspiration
            t_plus_237 = t + 237.3
            t_plus_273 = t + 273.2
            exp_term = torch.exp(17.3 * t / t_plus_237)
            pet = 29.8 * (dayl * 24) * 0.611 * exp_term / t_plus_273
            
            batch_size, time_steps = inputs.shape[0], inputs.shape[1]
            
            # Parameter scaling
            X1 = torch.clamp(self.x1, 1/2000, 1.0) * 2000
            X2 = self.x2 * 40 - 20
            X3 = torch.clamp(self.x3, 1/300, 1.0) * 300
            
            # Initialize states
            s = torch.zeros((batch_size, 1), dtype=torch.float32, device=inputs.device) + 0.3 * X1
            r = torch.zeros((batch_size, 1), dtype=torch.float32, device=inputs.device) + 0.7 * X3
            
            # Store outputs for all time steps
            qsim = []
            
            # Store routing inputs
            uh1_inputs = []
            uh2_inputs = []
            
            # Time step loop
            for t_idx in range(time_steps):
                P = p[:, t_idx].unsqueeze(-1)
                E = pet[:, t_idx].unsqueeze(-1)
                
                # Calculate net precipitation
                cond = P >= E
                Pn = torch.where(cond, P - E, torch.zeros_like(P))
                En = torch.where(cond, torch.zeros_like(E), E - P)
                
                # Calculate runoff generation
                Ps = torch.where(cond,
                    (X1 * (1 - (s / X1)**2) * torch.tanh(Pn / X1)) / (1 + (s / X1) * torch.tanh(Pn / X1)),
                    torch.zeros_like(P))
                Es = torch.where(~cond,
                    (s * (2 - s / X1) * torch.tanh(En / X1)) / (1 + (1 - s / X1) * torch.tanh(En / X1)),
                    torch.zeros_like(E))
                
                # Update states
                s = torch.clamp(s + Ps - Es, min=0.0, max=float(X1))
                Perc = s * (1 - torch.pow(1 + (4.0 / 9.0 * s / X1) ** 4, -0.25))
                s = torch.clamp(s - Perc, min=0.0, max=float(X1))
                PR = Perc + (Pn - Ps)
                
                # Calculate runoff
                uh1_input = 0.9 * PR
                uh2_input = 0.1 * PR
                
                # Store routing inputs
                uh1_inputs.append(uh1_input)
                uh2_inputs.append(uh2_input)
                
                # Calculate groundwater exchange
                F = X2 * torch.pow(r / X3, 3.5)
                r = torch.clamp(r + F, min=0.0)
                Qr = r * (1 - torch.pow(1 + (r / X3) ** 4, -0.25))
                r = torch.clamp(r - Qr, min=0.0, max=float(X3))
                
                # Calculate total flow
                Qt = Qr
                qsim.append(Qt)
            
            # Convert routing inputs to tensor
            uh1_inputs = torch.cat(uh1_inputs, dim=1).unsqueeze(1)  # [batch_size, 1, time_steps]
            uh2_inputs = torch.cat(uh2_inputs, dim=1).unsqueeze(1)  # [batch_size, 1, time_steps]
            
            # Use convolution for routing calculation
            uh1_contrib = self.uh1_conv(uh1_inputs).squeeze(1)  # [batch_size, time_steps]
            uh2_contrib = self.uh2_conv(uh2_inputs).squeeze(1)  # [batch_size, time_steps]
            
            # Add routing contributions to total flow
            q_sim = torch.stack(qsim, dim=1)  # [batch_size, time_steps, 1]
            q_sim = q_sim.squeeze(-1)  # [batch_size, time_steps]
            q_sim = q_sim + uh1_contrib + uh2_contrib
            q_sim = q_sim.unsqueeze(-1)  # [batch_size, time_steps, 1]
            
            return q_sim
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise

class ConvLayer(nn.Module):
    """
    One-dimensional convolutional layer
    Implements standard 1D-CNN operations
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: str = 'causal'):
        """Initialize convolutional layer
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            padding: Padding mode, supports 'causal' or 'same'
        """
        super(ConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Define convolutional layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor, shape [batch_size, channels, time_steps]
            
        Returns:
            torch.Tensor: Output tensor, shape [batch_size, out_channels, time_steps]
        """
        try:
            # Ensure input is float32 type
            x = x.to(torch.float32)
            
            # Handle causal padding
            if self.padding == 'causal':
                x = F.pad(x, (self.kernel_size - 1, 0))
            
            # Apply convolution and activation function
            x = self.conv(x)
            x = F.elu(x)
            return x
        except Exception as e:
            logger.error(f"Convolutional layer forward pass failed: {str(e)}")
            raise

class ScaleLayer(nn.Module):
    """
    Data standardization layer
    Function: Standardizes input meteorological data (mean 0, std 1), keeps flow data unchanged
    """
    def __init__(self):
        super(ScaleLayer, self).__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor, shape [batch_size, time_steps, features]
               features include meteorological data and flow data
            
        Returns:
            torch.Tensor: Standardized tensor, same shape as input
        """
        try:
            # Ensure input is float32 type
            x = x.to(torch.float32)
            
            # Split input data
            met = x[:, :, :-1]   # Meteorological data
            flow = x[:, :, -1:]  # Flow data
            
            # Calculate statistical features
            met_mean = torch.mean(met, dim=-2, keepdim=True)
            met_std = torch.std(met, dim=-2, keepdim=True)
            
            # Standardize
            met_scaled = (met - met_mean) / (met_std + 1e-6)
            
            # Recombine
            return torch.cat([met_scaled, flow], dim=-1)
        except Exception as e:
            logger.error(f"Standardization layer forward pass failed: {str(e)}")
            raise 