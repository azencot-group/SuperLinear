from typing import Optional, Tuple, Dict, List, Union
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .configuration_super_linear import SuperLinearConfig


"-------------------------------------------------------------------------------------------------------------------"
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, norm_type=None, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.norm_type = norm_type
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))

        if self.subtract_last:
            self.last = x[:, -1:, :].detach()
            self.mean = torch.mean(x[:, :-1, :], dim=dim2reduce, keepdim=True).detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        
        if self.norm_type == "l1":
            self.stdev = torch.mean(torch.abs(x - self.mean), dim=dim2reduce, keepdim=True).detach()
        elif self.norm_type == "l2":
            self.stdev = torch.sqrt(torch.mean((x - self.mean) ** 2, dim=dim2reduce, keepdim=True) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev

        if self.norm_type in ["l1", "l2"]:
            x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
            
        if self.norm_type in ["l1", "l2"]:
            x = x * self.stdev
            
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        
        return x
"-------------------------------------------------------------------------------------------------------------------"
class Linear(nn.Module):
    """Simple linear layer expert."""
    def __init__(self, input_len, output_len):
        super(Linear, self).__init__()
        self.Linear = nn.Linear(input_len, output_len)

    def forward(self, x):
        # x: [Batch*Channel, Input length]
        x = x.clone()
        x = self.Linear(x).clone()
        return x # to [Batch, Output length, Channel]   
    
class Naive(nn.Module):
    """Naive forecasting expert - repeats last value."""
    def __init__(self, input_len, output_len):
        super(Naive, self).__init__()
        self.output_len = output_len

    def forward(self, x):
        # x: [Batch*Channel, Input length]
        x = x[:,-1].unsqueeze(1).repeat(1, self.output_len)
        return x # to [Batch, Output length, Channel]   
    
class Mean(nn.Module):
    """Mean forecasting expert - repeats mean value."""
    def __init__(self, input_len, output_len):
        super(Mean, self).__init__()
        self.output_len = output_len

    def forward(self, x):
        # x: [Batch*Channel, Input length]
        x = x.mean(dim=1).unsqueeze(1).repeat(1, self.output_len)
        return x # to [Batch, Output length, Channel]  

class RLinear(nn.Module):
    """Reversible Instance Normalization Linear layer expert."""
    def __init__(self, input_len, output_len):
        super(RLinear, self).__init__()
        self.Linear = nn.Linear(input_len, output_len)
        self.revin_layer = RevIN(num_features = None, affine=False, norm_type = None, subtract_last = False)

    def forward(self, x):
        # x: [Batch, Input length,Channel]
        x_shape = x.shape
        if len(x_shape) == 2:
            x = x.unsqueeze(-1)
        x = x.clone()
        x = self.revin_layer(x, 'norm')
        
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1).clone()
        x = self.revin_layer(x, 'denorm')
        if len(x_shape) == 2:
            x = x.squeeze(-1)
        return x # to [Batch, Output length, Channel]  

"-------------------------------------------------------------------------------------------------------------------"
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
        x_0 = inputs - torch.mean(inputs, dim=1, keepdim=True)  # Remove mean (DC component)

        # Compute FFT and normalize
        dft = torch.fft.fft(x_0, dim=1, n=n) / np.sqrt(n)
        dft = dft[:, :n//2]  # Keep only positive frequencies
        I = torch.abs(dft) ** 2  # Power spectral density

        # Normalize periodogram
        I_sum = torch.sum(I, dim=1, keepdim=True)
        I_sum[I_sum == 0] = 1  # Avoid division by zero
        I = I / I_sum
            
        return I

    def forward(self, x, get_prob=False, get_prob_only=False):
        """
        Forward pass through the Mixture of Experts.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length]
            get_prob: Whether to return expert selection probabilities
            get_prob_only: Whether to return only probabilities without computation
            
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
        gate_outputs = self.gating_network(x_0)  # Raw gating scores
        
        if self.moe_norm:
            gate_outputs = self.batch_norm(gate_outputs)

        # Apply temperature scaling during inference
        if not self.training:
            gate_outputs = gate_outputs / self.moe_temp

        if get_prob_only:
            expert_probs = F.softmax(gate_outputs, dim=1)
            return expert_probs

        # Add noise to gating logits during training (for exploration)
        if self.training:
            noise = torch.randn_like(gate_outputs).to(x.device) * self.noise_std
            noisy_gate_outputs = gate_outputs + noise
            topk_values, topk_indices = torch.topk(noisy_gate_outputs, self.k, dim=1)
        else:
            topk_values, topk_indices = torch.topk(gate_outputs, self.k, dim=1)

        # Normalize the gate values with softmax
        topk_gates = F.softmax(topk_values, dim=1)
    
        # Get outputs from all experts
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.num_experts)], dim=1)

        # Select only the outputs from the top-k experts
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(2))
        sparse_expert_outputs = torch.gather(expert_outputs, 1, topk_indices_expanded)

        # Combine expert outputs using the gate values
        output = torch.sum(topk_gates.unsqueeze(2) * sparse_expert_outputs, dim=1)
        
        if get_prob:
            expert_probs = F.softmax(gate_outputs, dim=1)
            return output, expert_probs
        
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

        self.configs = copy.deepcopy(configs)

        # Core model configuration
        self.train_pred_len = configs.train_pred_len
        self.train_seq_len = configs.train_seq_len
        self.layer_type = configs.layer_type


        # Initialize additional configuration attributes with defaults
        self.long_horizon_scaling = configs.long_horizon_scaling
        self.lookback_resampling = configs.lookback_resampling
        lookback_scale_str = configs.scale_list
        if isinstance(lookback_scale_str, str):
            self.scale_list = [float(x.strip()) for x in lookback_scale_str.split(',')]
        else:
            self.scale_list = lookback_scale_str  # Already a list
        self.threshold = configs.threshold
        self.freq_bound = configs.freq_bound
        self.penalty_scale = configs.penalty_scale
        self.fft_len = configs.fft_len

        # Parse frequency experts from configuration
        freq_experts_str = configs.freq_experts
        if freq_experts_str == "":
            self.freq_experts = None
        else:
            self.freq_experts = freq_experts_str.split('_')

        # Expert configuration
        self.top_k_experts = configs.top_k_experts
        self.freeze_experts = configs.freeze_experts

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
            self.n_experts = len(self.experts)
        else:
            raise ValueError("Please specify experts in the configuration.")

        # Create additional complementary experts if specified
        comp_moe = configs.comp_moe
        if comp_moe > 0:
            if comp_moe == 1:
                print("Creating complementary expert")
                self.experts["comp"] = RLinear(self.train_seq_len, self.train_pred_len)
            else:
                for i in range(comp_moe):
                    print(f"Creating complementary expert {i}")
                    self.experts["comp_"+str(i)] = RLinear(self.train_seq_len, self.train_pred_len)
                    
        # Initialize the MoE layer and dropout    
        self.moe = SparseMoE(configs, experts=self.experts.values())
            
        print("Experts:", self.experts.keys())

    def add_experts(self, experts: Dict[str, nn.Module]) -> nn.Module:
        """
        Add new experts to the model.
        
        Args:
            experts: Dictionary of expert instances to add
            
        Returns:
            Updated MoE layer
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

    def apply_long_horizon_scaling(self, ar_out: torch.Tensor, ar_x: torch.Tensor) -> torch.Tensor:
        """
        Apply scaling to auto-regressive outputs to maintain statistical properties during long horizon prediction.
        
        This function identifies cases where the variance of the new predictions exceeds the variance
        of the input sequence and applies scaling to maintain consistent statistical properties.
        
        Args:
            ar_out: Auto-regressive output tensor of shape [batch_size * features, pred_len]
            ar_x: Input sequence tensor of shape [batch_size * features, seq_len]
            
        Returns:
            Scaled auto-regressive output tensor
        """
        if not (self.long_horizon_scaling and not self.training):
            return ar_out
            
        # Calculate statistics for scaling
        std_new = torch.std(ar_out, dim=1, keepdim=True)
        mean_new = torch.mean(ar_out, dim=1, keepdim=True)
        std_old = torch.std(ar_x, dim=1, keepdim=True)
        
        # Find indices where new variance exceeds old variance
        inds = torch.where(std_new / std_old > 1)[0]
        
        if len(inds) > 0:
            # Center the outputs around their mean
            ar_out_centered = ar_out[inds] - mean_new[inds]
            
            # Calculate scaling factor to match old variance
            scaling = std_old[inds] / (std_new[inds] + 1e-8)
            
            # Scale and shift back to mean_new
            ar_out_adjusted = ar_out_centered * scaling + mean_new[inds]
            ar_out[inds] = ar_out_adjusted
            
        return ar_out

    def lookback_resample_search(self, x, scale_list=[2,4,6], min_lookback=512):
        """
        Search for optimal resampling scale based on lookback analysis of expert selection.

        This function analyzes the frequency content and expert selection lookback to determine
        the best resampling scale for each input sequence, potentially improving model performance
        by matching input characteristics to expert capabilities.
        
        Args:
            x: Input tensor of shape [batch_size, features, sequence_length]
            scale_list: List of potential downsampling scales to evaluate
            min_lookback: Minimum sequence length required after resampling
            
        Returns:
            Tuple of (resampled_input, final_scales) where:
            - resampled_input: Optimally resampled input tensor
            - final_scales: Scale factors used for each sample
        """
        B, V, L = x.shape

        lookback = self.train_seq_len
        x_0 = x.reshape(B*V, L)[:, -lookback:]
        output_x = x_0.clone()[:, -lookback:]

        x_reshape = x.reshape(B*V, L)
        x_fft_init = self.moe.get_periodogram(x_reshape, n=self.fft_len)

        right_cumsum = torch.cumsum(x_fft_init, dim=-1)
        mask = right_cumsum > 1-self.threshold
        j_threshold = mask.float().argmax(dim=-1)

        freqs = np.array([np.linspace(0, 0.5, self.fft_len//2)])
        threshhold_freqs = np.take_along_axis(freqs, j_threshold.unsqueeze(-1).detach().cpu().numpy(), axis=1)
        
        # where threshhold_freqs is 0, set to a small value to avoid division by zero
        threshhold_freqs[threshhold_freqs == 0] = self.freq_bound
        max_scale_factor = (self.freq_bound/ threshhold_freqs).astype(int).flatten()


        if self.threshold==0:
            max_scale_factor = np.inf * np.ones(B*V, dtype=int)

        # Compute energy loss penalty for each potential scale
        energy_loss_penalties = {}
        total_energy = torch.sum(x_fft_init, dim=-1)  # Total energy per sample
        
        for scale in scale_list:
            if scale <= 1:
                continue  # No penalty for upsampling or no scaling
                
            # Calculate Nyquist frequency after downsampling
            nyquist_after_downsample = 0.5 / scale
            
            # Find frequency bins that will be lost (above new Nyquist)
            freq_bins = torch.linspace(0, 0.5, self.fft_len//2, device=x_fft_init.device)
            lost_freq_mask = freq_bins > nyquist_after_downsample
            
            # Calculate energy that will be lost
            lost_energy = torch.sum(x_fft_init[:, lost_freq_mask], dim=-1)
            # Energy loss fraction (0 = no loss, 1 = all energy lost)
            energy_loss_fraction = lost_energy / (total_energy + 1e-10)
            energy_loss_penalties[scale] = energy_loss_fraction

        # Get initial entropy
        prob = self.moe(x_0, get_prob_only=True)
        best_scores = -torch.sum(prob * torch.log(prob + 1e-10), dim=-1)
        final_scales = torch.ones(B*V, device=x.device)

        for scale in scale_list:
            x_interp = torch.nn.functional.interpolate(
                x, scale_factor=1/scale, mode='linear', align_corners=True
            )
            
            if x_interp.shape[2] >= min_lookback:
                x_interp_reshaped = x_interp.reshape(B*V, x_interp.shape[-1])
                x_interp_reshaped = x_interp_reshaped[:, -lookback:]
                prob = self.moe(x_interp_reshaped, get_prob_only=True)

                scores = -torch.sum(prob * torch.log(prob + 1e-10), dim=-1)
                
                # Add energy loss penalty
                if scale in energy_loss_penalties:
                    energy_penalty = energy_loss_penalties[scale]
                    scores = scores + energy_penalty*self.penalty_scale

                idx = np.where((scores < best_scores).cpu() & torch.tensor(max_scale_factor >= scale))[0]

                if len(idx) > 0:
                    output_x[idx] = x_interp_reshaped[idx]
                    best_scores[idx] = scores[idx]
                    final_scales[idx] = scale

        return output_x.reshape(B, V, output_x.shape[-1]), final_scales

    def lookback_resample_reverse(self, y, final_scales, inf_pred_len=None):
        """
        Reverse the resampling operation on the output.
        
        This function upsamples the model outputs back to the original scale
        based on the resampling factors used during input processing.
        
        Args:
            y: Output tensor from model of shape [batch_size, features, pred_len]
            final_scales: Scale factors used during input resampling
            inf_pred_len: Target prediction length
            
        Returns:
            Upsampled output tensor of shape [batch_size, features, inf_pred_len]
        """
        B, V, L = y.shape
        y_reshaped = y.view(B*V, L)
        y_out = y_reshaped[:, :inf_pred_len]

        unique_scales = torch.unique(final_scales)
        for scale in unique_scales:
            scale_val = scale.item()  # Convert tensor to scalar
            if scale_val > 1:
                idx = torch.where(final_scales == scale)[0]

                if len(idx) > 0:
                    y_interp = torch.nn.functional.interpolate(
                        y_reshaped[idx].unsqueeze(1), scale_factor=scale_val, mode='linear', align_corners=True
                    )
                    y_out[idx] = y_interp.reshape(len(idx), y_interp.shape[-1])[:, :inf_pred_len]
        return y_out.reshape(B, V, inf_pred_len)

    def forward(self, x_in: torch.Tensor, get_prob: bool = False, pred_len: Optional[int] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x_in: Encoder input tensor of shape [batch_size, sequence_length] or [batch_size, features, sequence_length]
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
            x = x.unsqueeze(1)

        B, V, L = x.shape

        short_lookback = False
        orig_pred_len = pred_len
        
        if L < self.train_seq_len:
            # Handle case where input sequence is shorter than expected
            # by interpolating to the required length
            scale_factor = self.train_seq_len / L
            scale_factor = int(np.ceil(scale_factor))

            pred_len = pred_len * scale_factor
            x = interpolate(x, scale_factor=scale_factor, mode='linear')

            x = x[:, :, -self.train_seq_len:]
            L = self.train_seq_len

            short_lookback = True

        # lookback resampling logic
        final_scales = None
        
        if self.lookback_resampling and L > self.train_seq_len:

            x_resampled, final_scales = self.lookback_resample_search(
                x, self.scale_list, self.train_seq_len
            )
            
            # Update x and L for the resampled input
            x = x_resampled
            L = x.shape[-1]


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
                ar_out = self.moe(ar_x)
                ar_out = self.apply_long_horizon_scaling(ar_out, ar_x)
                outputs.append(ar_out)
                ar_x = torch.cat([ar_x, ar_out], dim=1)[:, -self.train_seq_len:]
            out = torch.cat(outputs, dim=1)[:, :pred_len]

        # Reshape back to batch format
        out = out.reshape(B, V, out.shape[-1])

        # Apply lookback resampling reverse if it was used
        if self.lookback_resampling and final_scales is not None and not short_lookback:
            out = self.lookback_resample_reverse(out, final_scales, orig_pred_len)

        # If we used interpolation earlier, now downsample back to original scale
        if short_lookback:
            out = interpolate(out, scale_factor=1/scale_factor, mode='linear')
        out = out[:, :, :orig_pred_len]

            
        if x_in.dim() == 2:
            out = out.squeeze(1)
        
        if get_prob:
            expert_probs = expert_probs.reshape(B, V, expert_probs.shape[-1])
          #  expert_probs = expert_probs.permute(0, 2, 1)  # [batch_size, num_experts, sequence_length]
            if x_in.dim() == 2:
                expert_probs = expert_probs.squeeze(-1)
            return out, expert_probs

        return out

    def map_to_cycle(self, freq: str) -> int:
        """
        Map frequency string notation to cycle length (number of periods).
        
        Args:
            freq: String representing a time frequency (e.g., "h" for hourly, "D" for daily)
            
        Returns:
            Integer representing the number of periods in the cycle
        """
        cycle = int(freq.split("/")[1])
        return cycle

"-------------------------------------------------------------------------------------------------------------------"
class SuperLinearForCausalLM(PreTrainedModel):
    config_class = SuperLinearConfig

    def __init__(self, config: SuperLinearConfig):
        super().__init__(config)
        
        # the backbone keeps its own Config dataclass, so build one on-the-fly:
        backbone_cfg = type("Cfg", (), config.to_dict())()
        self.args = backbone_cfg
        self.backbone = Model(backbone_cfg)
        self.post_init()

    # ------------------------------------------------------------------
    # Forward pass expected by AutoModelForCausalLM
    # ------------------------------------------------------------------
    def forward(self,
                inputs_embeds: torch.Tensor = None,               
                pred_len: Optional[int] = None,
                get_prob: bool = False,
                **kwargs) -> CausalLMOutputWithCrossAttentions:

        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided")
        
        # backbone expects (B, C, L) or (B, L) 
        x_enc = inputs_embeds
    
        # backbone returns (B, pred_len, C)
        if get_prob:
            preds, probs = self.backbone(x_enc, pred_len=pred_len, get_prob=True)
        else:
            preds = self.backbone(x_enc, pred_len=pred_len, get_prob=False)
            probs = None
            
        return CausalLMOutputWithCrossAttentions(
            logits=preds,
            hidden_states=None,
            attentions=probs
        )





