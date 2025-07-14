from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .configuration_super_linear import SuperLinearConfig


"-------------------------------------------------------------------------------------------------------------------"
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, norm_type = None, subtract_last = False):
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

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))

        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        if  self.norm_type == "l1":
            self.denom = torch.sum(torch.abs(x), dim=dim2reduce, keepdim=True).detach()
        elif  self.norm_type == "l2":
            self.denom = torch.sqrt(torch.sum(x**2, dim=dim2reduce, keepdim=True)).detach()

            
    def _normalize(self, x):

        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev

        if self.norm_type in ["l1", "l2"]:
            x = x / self.denom

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        if self.norm_type in ["l1", "l2"]:
            x = x * self.denom
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
            self.gate_norm = nn.BatchNorm1d(self.num_experts)

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
            self.gate_outputs = self.gate_norm(self.gate_outputs)

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
        self.configs = configs
        self.model_name = "SuperLinear"
        self.train_pred_len = configs.train_pred_len
        self.train_seq_len = configs.train_seq_len
        self.resample_long_lookback = configs.resample_long_lookback
        self.layer_type = configs.layer_type

        # Parse frequency experts from configuration
        if configs.freq_experts == "":
            self.freq_experts = None
        else:
            self.freq_experts = configs.freq_experts.split('_')

        self.top_k_experts = configs.top_k_experts
        self.freeze_experts = configs.freeze_experts

        # Initialize experts based on frequency specification or create generic experts
        self.experts = {}
        if self.freq_experts is not None:
            for expert_freq in self.freq_experts:
                if expert_freq == "naive" or expert_freq == "Naive":
                    self.experts[expert_freq] = Naive(self.train_seq_len, self.train_pred_len)
                elif expert_freq == "mean" or expert_freq == "Mean":    
                    self.experts[expert_freq] = Mean(self.train_seq_len, self.train_pred_len)
                else:
                    # Use the appropriate expert class based on layer_type
                    expert_classes = {'Linear': Linear, 'RLinear': RLinear}
                    if self.layer_type in expert_classes:
                        expert_class = expert_classes[self.layer_type]
                        self.experts[expert_freq] = expert_class(self.train_seq_len, self.train_pred_len)
                    else:
                        # Default to RLinear if unknown layer type
                        self.experts[expert_freq] = RLinear(self.train_seq_len, self.train_pred_len)
        else:
            raise ValueError("No frequency experts specified in configuration.")

        # Create additional complementary experts if specified
        if configs.comp_moe > 0:
            for i in range(configs.comp_moe):
                expert_classes = {'Linear': Linear, 'RLinear': RLinear}
                if self.layer_type in expert_classes:
                    expert_class = expert_classes[self.layer_type]
                    self.experts[f"comp_{i}"] = expert_class(self.train_seq_len, self.train_pred_len)
                else:
                    # Default to RLinear if unknown layer type
                    self.experts[f"comp_{i}"] = RLinear(self.train_seq_len, self.train_pred_len)
                    
        # Initialize the MoE layer    
        self.moe = SparseMoE(configs, experts=self.experts.values())
            
        print("Experts:", self.experts.keys())

    def add_experts(self, experts: dict):
        """
        Add new experts to the model.
        
        Args:
            experts: Dictionary of expert instances to add
        """
        for name, expert in experts.items():
            self.experts[name] = expert
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
            orig_pred_len: Original prediction length (required for inverse=True)
            
        Returns:
            Tuple of (resampled_tensor, updated_pred_len, scale_factor, orig_pred_len)
            For inverse=True: returns (resampled_tensor, None, None, None)
        """
        if not inverse:
            # Upsample if input is shorter than training length
            if x.size(-1) < self.train_seq_len:
                scale_factor = self.train_seq_len / x.size(-1)
                x_resampled = F.interpolate(x.unsqueeze(1), size=self.train_seq_len, mode='linear', align_corners=False).squeeze(1)
                pred_len_resampled = int(pred_len * scale_factor)
                return x_resampled, pred_len_resampled, scale_factor, pred_len
            else:
                return x, pred_len, None, None
        else:
            # Downsample back to original scale
            if orig_pred_len is not None:
                x_resampled = F.interpolate(x.unsqueeze(1), size=orig_pred_len, mode='linear', align_corners=False).squeeze(1)
                return x_resampled, None, None, None
            else:
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

        # Handle resampling if input is shorter than training length
        if self.resample_long_lookback and L < self.train_seq_len:
            x, pred_len, scale_factor, orig_pred_len = self.resample_seq_len(x, pred_len, inverse=False)

        # Reshape for MoE processing
        x = x.reshape(B * V, x.size(-1))

        # Forward through MoE
        if get_prob:
            out, expert_probs = self.moe(x, get_prob=True)
        else:
            out = self.moe(x)

        # Reshape back
        out = out.reshape(B, V, out.size(-1))

        # Handle resampling back to original scale if needed
        if scale_factor is not None:
            out, _, _, _ = self.resample_seq_len(out, None, inverse=True, orig_pred_len=orig_pred_len)

        # Return to original shape conventions
        result = out.permute(0, 2, 1)
        
        if x_in.dim() == 2:
            result = result.squeeze(-1)

        if get_prob:
            expert_probs = expert_probs.reshape(B, V, expert_probs.shape[-1])
            return result, expert_probs
        return result
"-------------------------------------------------------------------------------------------------------------------"
class SuperLinearForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SuperLinearConfig

    def __init__(self, config: SuperLinearConfig):
        super().__init__(config)
  

        # the backbone keeps its own Config dataclass, so build one on‑the‑fly:
        backbone_cfg   = type("Cfg", (), config.to_dict())()
        self.args      = backbone_cfg
        self.backbone  = Model(backbone_cfg)
        self.post_init()                             

    # ------------------------------------------------------------------
    # Forward pass expected by AutoModelForCausalLM
    # ------------------------------------------------------------------
    def forward(self,
                inputs_embeds: torch.Tensor = None,           
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple] = None,
                use_cache: bool = True,
                labels: Optional[torch.Tensor] = None,        
                **kwargs,) -> CausalLMOutputWithCrossAttentions:


        if inputs_embeds is None:
            raise ValueError("Pass the time‑series as `inputs_embeds`")
        
        # backbone expects (B, C, L)
        x_enc = inputs_embeds
    
        # backbone returns (B, pred_len, C)
        preds = self.backbone(x_enc)
        return CausalLMOutputWithCrossAttentions(loss=None,logits=preds,past_key_values=None,hidden_states=None,attentions=None,)


    def prepare_inputs_for_generation(self, inputs_embeds, past_key_values=None, **kwargs):
        if past_key_values is not None:
            # only feed the last new step
            inputs_embeds = inputs_embeds[:, -1:, :]
        return {"inputs_embeds": inputs_embeds, "past_key_values": past_key_values}

    def _reorder_cache(self, past, beam_idx, **kwargs):
        return past  # backbone keeps no KV cache

