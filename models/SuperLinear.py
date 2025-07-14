from layers.Linear_layers import RLinear, Naive, Mean
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.nn.functional import interpolate


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts (MoE) module that routes inputs to the most relevant experts.
    
    This implementation uses a gating network to determine which experts should process each input.
    Only the top-k experts are used for each input, creating a sparse computation pattern.
    
    Args:
        configs: Configuration object containing MoE parameters
        experts: Collection of expert modules (neural networks)
    """
    def __init__(self, configs, experts=None):
        super(SparseMoE, self).__init__()
        self.noise_std = configs.noisy_gating_std
        self.experts = nn.ModuleList(experts)  # Store experts in ModuleList for proper registration
        self.num_experts = len(experts)
        self.k = configs.top_k_experts
        
        if self.k > self.num_experts:
            print(f"Warning: k ({self.k}) is greater than the number of experts ({self.num_experts}). Setting k to {self.num_experts}.")
            self.k = self.num_experts
            
        self.moe_temp = configs.moe_temp
        self.use_fft = configs.use_fft
        self.fft_len = configs.fft_len
        self.moe_norm = configs.moe_norm
    
        
        # Initialize gating network based on configuration
        if self.use_fft:
            self.gating_network = nn.Linear(self.fft_len//2, self.num_experts, bias=True)
        else:
            self.gating_network = nn.Linear(configs.train_seq_len, self.num_experts, bias=True)

        if self.moe_norm:
            self.batch_norm = nn.BatchNorm1d(self.num_experts)

    def get_periodogram(self, inputs, n=10000):
        """
        Calculate the periodogram (power spectral density) of input time series.
        
        The periodogram is used as a frequency-domain representation of the signal
        to help the gating network identify periodic patterns.
        
        Args:
            inputs: Input time series tensor of shape [batch_size, sequence_length] or [batch_size, sequence_length, features]
            n: Number of points in FFT computation
            
        Returns:
            Normalized periodogram of the input signals
        """
        if inputs.dim() == 2:
            x_0 = inputs.unsqueeze(2)
        else:
            x_0 = inputs
        x_0 = x_0 - torch.mean(x_0, dim=1, keepdim=True)  # Remove mean (DC component)

        # Compute FFT and normalize
        dft = torch.fft.fft(x_0, dim=1, n=n) / np.sqrt(n)
        dft = dft[:, :n//2, :]  # Keep only positive frequencies
        I = torch.abs(dft) ** 2  # Power spectral density

        # Normalize periodogram
        I_sum = torch.sum(I, dim=1, keepdim=True)
        I_sum[I_sum == 0] = 1  # Avoid division by zero
        I = I / I_sum

        if torch.any(I_sum == 0):
            print("Zeros in the sum")
            raise ValueError

        if inputs.dim() == 2:
            I = I.squeeze(2)
            
        return I

    def forward(self, x, get_prob=False):
        """
        Forward pass through the Mixture of Experts.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length]
            get_prob: Whether to return expert selection probabilities
            
        Returns:
            - Output tensor from the selected experts
            - (Optional) Expert selection probabilities if get_prob is True
        """
        # Preprocess input if using FFT-based gating
        if self.use_fft:
           x_0 = self.get_periodogram(x, n=self.fft_len)
        else:
            x_0 = x
            
        # Get gating logits
        self.gate_outputs = self.gating_network(x_0)  # Raw gating scores
        
        if self.moe_norm:
            self.gate_outputs = self.batch_norm(self.gate_outputs)

        # Apply temperature scaling during inference
        if not self.training:
            self.gate_outputs = self.gate_outputs / self.moe_temp

        # Add noise to gating logits during training (for exploration)
        noise = torch.randn_like(self.gate_outputs).to(x.device) * self.noise_std
        if self.training:
            noisy_gate_outputs = self.gate_outputs + noise
            self.topk_values, topk_indices = torch.topk(noisy_gate_outputs, self.k, dim=1)
        else:
            self.topk_values, topk_indices = torch.topk(self.gate_outputs, self.k, dim=1)

        # Normalize the gate values with softmax
        self.topk_gates = F.softmax(self.topk_values, dim=1)
    
        batch_size = x.size(0)
        # Get outputs from all experts
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.num_experts)], dim=1)

        # Select only the outputs from the top-k experts
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(2))
        sparse_expert_outputs = torch.gather(expert_outputs, 1, topk_indices_expanded)

        # Combine expert outputs using the gate values
        output = torch.sum(self.topk_gates.unsqueeze(2) * sparse_expert_outputs, dim=1)
        
        if get_prob:
            expert_probs = F.softmax(self.gate_outputs, dim=1)
            return output,  expert_probs
        
        return output


class Model(nn.Module):
    """
    Main model class that employs a Mixture of Experts for time series forecasting.
    
    This model can work with various types of linear layers as experts and supports
    both standard prediction and auto-regressive prediction for longer horizons.
    
    Args:
        configs: Configuration object containing model parameters
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.model_name = "SuperLinear"
        self.train_pred_len = configs.train_pred_len
        self.train_seq_len = configs.train_seq_len
        self.resample_long_lookback = configs.resample_long_lookback
        self.layer_type = configs.layer_type
        self.load_weights_full = configs.load_weights_full
        self.load_linear = configs.load_linear

        if self.load_weights_full:
            self.load_linear = False  # If loading full weights, loading linear experts is not needed


        # Parse frequency experts from configuration
        if configs.freq_experts == "":
            self.freq_experts = None
        else:
            self.freq_experts = configs.freq_experts.split('_')


        self.top_k_experts = configs.top_k_experts
        self.freeze_experts = configs.freeze_experts
        path = configs.linear_freq_weights_path
        linear_freq_dirs = os.listdir(path)
        checkpoints_paths = [path + "/" + d + "/" + "checkpoint.pth" for d in linear_freq_dirs]


        # Initialize experts based on frequency specification or create generic experts
        self.experts = {}
        if self.freq_experts is not None:
            for expert_freq in self.freq_experts:
                if expert_freq.lower() == "naive":
                    self.experts[expert_freq] = Naive(self.train_seq_len, self.train_pred_len)
                elif expert_freq.lower() == "mean":    
                    self.experts[expert_freq] = Mean(self.train_seq_len, self.train_pred_len)
                else:
                    self.experts[expert_freq] = RLinear(self.train_seq_len, self.train_pred_len)
                    if configs.load_linear:
                        cycle = self.map_to_cycle(expert_freq)
                        cycle_str = f'cycle_{cycle}/'
                        cycle_checkpoint_path = [cp for cp in checkpoints_paths if (cycle_str in cp and self.layer_type in cp)]
                        if len(cycle_checkpoint_path) > 0:
                            cycle_checkpoint_path = cycle_checkpoint_path[0]
                            self.experts[expert_freq].load_state_dict(torch.load(cycle_checkpoint_path))
                        else:
                            print(f"Checkpoint for {cycle_str} not found in {path}")
                            raise ValueError(f"Checkpoint for {cycle_str} not found in {path}")
                        
                        if configs.freeze_experts:
                            for param in self.experts[expert_freq].parameters():
                                param.requires_grad = False
                        
            self.n_experts = len(self.experts)
        else:
            for i in range(self.n_experts):
                print(f"creating expert {i}")
                self.experts[str(i)] = RLinear(self.train_seq_len, self.train_pred_len)

        # Create additional complementary experts if specified
        if configs.comp_moe > 0:
            if configs.comp_moe == 1:
                print("Creating complementary expert")
                self.experts["comp"] = RLinear(self.train_seq_len, self.train_pred_len)
            else:
                for i in range(configs.comp_moe):
                    print(f"Creating complementary expert {i}")
                    self.experts["comp_"+str(i)] = RLinear(self.train_seq_len, self.train_pred_len)
                    
        # Initialize the MoE layer and dropout    
        self.moe = SparseMoE(configs, experts=self.experts.values())

        # Load pre-trained weights if specified
        if configs.load_weights_full:
            path = configs.full_weights_path
            print(f"Loading weights from {path}")
            if os.path.exists(path):
                checkpoint = torch.load(path)
                self.load_state_dict(checkpoint)
            else:   
                raise ValueError(f"Checkpoint {path} does not exist. Please check the path.")
        print("Experts:", self.experts.keys())


    def add_experts(self, experts: dict):
            """
            Add new experts to the model.
            
            Args:
                experts: Dictionary of expert instances to add
            """
            for name, expert in experts.items():
                if name not in self.experts:
                    self.experts[name] = expert
                    print(f"Added expert: {name}")
                else:
                    print(f"Expert {name} already exists. Skipping addition.")
            # Reinitialize the MoE layer with the updated experts
            self.moe = SparseMoE(self.configs, experts=self.experts.values())
            return self.moe

    def resample_seq_len(self, x, pred_len, inverse=False, orig_pred_len=None):
        """
        Resample sequence length for handling inputs shorter than expected training length.
        
        Args:
            x: Input tensor
            pred_len: Prediction length
            inverse: If True, downsample back to original scale; if False, upsample
            scale_factor: Scale factor for interpolation (required for inverse=True)
            orig_pred_len: Original prediction length (required for inverse=True)
            
        Returns:
            Tuple of (resampled_tensor, updated_pred_len, scale_factor, orig_pred_len)
            For inverse=True: returns (resampled_tensor, None, None, None)
        """
        if not inverse:
            # Upsampling: interpolate to required length
            B, V, L = x.shape
            scale_factor = self.train_seq_len / L
            scale_factor = int(np.ceil(scale_factor))
            orig_pred_len = pred_len
            
            pred_len = pred_len * scale_factor
            x = interpolate(x, scale_factor=scale_factor, mode='linear')
            return x, pred_len, scale_factor, orig_pred_len
        else:
            # Downsampling: interpolate back to original scale
            if scale_factor is None or orig_pred_len is None:
                raise ValueError("scale_factor and orig_pred_len must be provided for inverse resampling")
            
            x = interpolate(x, scale_factor=1/scale_factor, mode='linear')
            x = x[:, :, :orig_pred_len]
            
            return x, None, None, None

    def forward(self, x_in, get_prob=False, pred_len=None):
        """
        Forward pass through the model.
        
        Args:
            x_in: Encoder input tensor
            get_prob: Whether to return expert selection probabilities
            pred_len: Override for prediction length
            
        Returns:
            - Prediction tensor
            - (Optional) Expert selection probabilities if get_prob is True
        """
        if pred_len is None:
            pred_len = self.train_pred_len

        x = x_in
        # If input is 2D, add a channel dimension
        if x_in.dim() == 2:
            x = x.unsqueeze(-1)

        # Permute to shape [batch_size, features, sequence_length]
        x = x.permute(0, 2, 1)
        B, V, L = x.shape


        scale_factor = None
        orig_pred_len = None
        short_lookback = L < self.train_seq_len
        long_lookback = L > self.train_seq_len

        
        if short_lookback or (self.resample_long_lookback and long_lookback):
            # Handle case where input sequence is shorter than expected
            x, pred_len, scale_factor, orig_pred_len = self.resample_seq_len(x, pred_len, inverse=False)
            L = self.train_seq_len

        x = x[:, :, -self.train_seq_len:]  # Truncate to training sequence length
        
        # Reshape to process each feature independently
        x = x.reshape(B * V, L)
        expert_probs = None
        
        # Forward pass through MoE
        if get_prob:
            out, expert_probs = self.moe(x, get_prob=True)
        else:
            out = self.moe(x)

        # Auto-regressive prediction for long horizons
        if self.train_pred_len < pred_len:
            outputs = [out]
            ar_x = torch.cat([x, out], dim=1)[:, -self.train_seq_len:]
            for i in range(0, pred_len, self.train_pred_len):
                ar_out, _ = self.moe(ar_x)
                outputs.append(ar_out)
                ar_x = torch.cat([ar_x, ar_out], dim=1)[:, -self.train_seq_len:]
            out = torch.cat(outputs, dim=1)[:, :pred_len]
            
        # Reshape back to batch format
        out = out.reshape(B, V, out.shape[-1])
        
        # If we used interpolation earlier, now downsample back to original scale
        if short_lookback or self.resample_long_lookback:
            out, _, _, _ = self.resample_seq_len(out, pred_len, inverse=True, scale_factor=scale_factor, orig_pred_len=orig_pred_len)
            
        # Final permutation to expected output format [batch_size, pred_length, features]
        result = out.permute(0, 2, 1)

        if x_in.dim() == 2:
            result = result.squeeze(-1)
        
        if get_prob:
            expert_probs = expert_probs.reshape(B, V, expert_probs.shape[-1])
            expert_probs = expert_probs.permute(0, 2, 1)  # [batch_size, num_experts, sequence_length]
            if x_in.dim() == 2:
                expert_probs = expert_probs.squeeze(-1)
            return result, expert_probs
            
        return result
    
    def map_to_cycle(self, freq):
        """
        Map frequency string notation to cycle length (number of periods).
        
        Args:
            freq: String representing a time frequency (e.g., "h" for hourly, "D" for daily)
            
        Returns:
            Integer representing the number of periods in the cycle
        """
        if "/" in freq:
            cycle = int(freq.split("/")[1])
        elif "h" in freq:
            cycle = 24
        elif "2h" in freq:
            cycle = 12
        elif "3h" in freq:
            cycle = 8
        elif "4h" in freq:
            cycle = 6
        elif "D" in freq:
            cycle = 7
        elif "DM" in freq:
            cycle = 30
        elif "W" in freq:
            cycle = 52
        elif "M" in freq:
            cycle = 12
        elif "min" in freq:
            cycle = 1440
        elif "5min" in freq:
            cycle = 288
        elif "10min" in freq:
            cycle = 144
        elif "15min" in freq:
            cycle = 96
        elif "20min" in freq:
            cycle = 72
        elif "30min" in freq:
            cycle = 48
        elif "40min" in freq:
            cycle = 36
        elif "45min" in freq:
            cycle = 32
        elif "60min" in freq:
            cycle = 24
        else:
            cycle = int(freq)
        return cycle