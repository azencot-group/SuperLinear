from   typing import Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F

from transformers                          import (PreTrainedModel,GenerationMixin,AutoConfig,AutoModelForCausalLM,)
from transformers.modeling_outputs         import CausalLMOutputWithCrossAttentions
from .configuration_super_linear            import SuperLinearConfig


import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np


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
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    """
    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    """
    def forward(self, x):
        # x: [Batch, Input length]
        # padding on the both ends of time series
        front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.unsqueeze(1)).squeeze(1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean    
    

class DLinear(nn.Module):
    def __init__(self, input_len, output_len, kernel_size = 25):
        super(DLinear, self).__init__()
        self.seasonal = nn.Linear(input_len, output_len)
        self.trend = nn.Linear(input_len, output_len)
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.decompsition = series_decomp(kernel_size)

    def forward(self, x):
        # x: [Batch*Input length,Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_output = self.seasonal(seasonal_init)
        trend_output = self.trend(trend_init)
        x = seasonal_output + trend_output
        return x # to [Batch, Output length, Channel]    
    
class Linear(nn.Module):
    def __init__(self, input_len, output_len):
        super(Linear, self).__init__()
        self.Linear = nn.Linear(input_len, output_len)

    def forward(self, x):
        # x: [Batch*Channel, Input length]
        x = x.clone()
        x = self.Linear(x).clone()
        return x # to [Batch, Output length, Channel]   
    
class Naive(nn.Module):
    def __init__(self, input_len, output_len):
        super(Naive, self).__init__()
        self.output_len = output_len


    def forward(self, x):
        # x: [Batch*Channel, Input length]
        x =  x[:,-1].unsqueeze(1).repeat(1, self.output_len)
        return x # to [Batch, Output length, Channel]   
    
class Mean(nn.Module):
    def __init__(self, input_len, output_len):
        super(Mean, self).__init__()
        self.output_len = output_len

    def forward(self, x):
        # x: [Batch*Channel, Input length]
        x =  x.mean(dim=1).unsqueeze(1).repeat(1, self.output_len)
        return x # to [Batch, Output length, Channel]  
    

class NLinear(nn.Module):
    def __init__(self, input_len, output_len):
        super(NLinear, self).__init__()
        self.Linear = nn.Linear(input_len, output_len)

    def forward(self, x):
        # x: [Batch, Input length,Channel]
        seq_last = x[:,-1:].detach()
        x = x - seq_last
        x = self.Linear(x)
        return x+seq_last # to [Batch, Output length, Channel]   
    
 
class RLinear(nn.Module):
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
class SparseNoisyMoE(nn.Module):
    def __init__(self, configs, experts=None):
        super(SparseNoisyMoE, self).__init__()
        input_dim = configs.seq_len
        output_dim = configs.pred_len
        self.k = configs.top_k_experts
        self.noise_std = configs.noisy_gating_std
        self.noise_std_decay = configs.noisy_gating_std_decay
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.ker_len = configs.ker_len
        self.con = configs.con
        self.d_model = configs.d_model
        self.mlp_gating = configs.mlp_gating
        self.moe_temp = configs.moe_temp
        self.use_fft = configs.use_fft
        self.fft_len = configs.fft_len
        
        if self.use_fft:
            if self.mlp_gating:
                self.gating_network = nn.Sequential(
                    nn.Linear(self.fft_len//2, self.d_model),
                    nn.ReLU(),
                    nn.Linear(self.d_model, self.num_experts)
                )
            else:
                self.gating_network = nn.Linear(self.fft_len//2, self.num_experts, bias=True)
        else:
            self.gating_network = nn.Linear(input_dim, self.num_experts, bias=True)

    def get_periodogram(self, inputs, ker_len=50, con=1, n=10000):
        if inputs.dim() == 2:
            x_0 = inputs.unsqueeze(2)
        else:
            x_0 = inputs
        x_0 = x_0 - torch.mean(x_0, dim=1, keepdim=True)

        v = torch.arange(0, n) / n
        if con:
            if ker_len is None:
                ker_len = n // 4
                ker_len = min(ker_len, 50)

            x_0 = x_0.permute(0, 2, 1)
            ker = (torch.ones(1, 1, ker_len) / ker_len).to(x_0.device)
            x_c = F.conv1d(x_0, ker, padding="same")
            x_c[:, :, :ker_len // 2] = x_c[:, :, ker_len // 2:ker_len // 2 + 1]
            x_c[:, :, -ker_len // 2:] = x_c[:, :, -ker_len // 2 - 1:-ker_len // 2]
            x_0 = x_0 - x_c
            x_0 = x_0.permute(0, 2, 1)

        dft = torch.fft.fft(x_0, dim=1, n=n) / np.sqrt(n)
        dft = dft[:, :n//2, :]
        I = torch.abs(dft) ** 2

        I_sum = torch.sum(I, dim=1, keepdim=True)
        I_sum[I_sum == 0] = 1
        I = I / I_sum

        if torch.any(I_sum == 0):
            print("Zeros in the sum")
            raise ValueError

        if inputs.dim() == 2:
            I = I.squeeze(2)
            
        return I

    def forward(self, x, get_prob=False):
        if self.use_fft:
            x_0 = self.get_periodogram(x, ker_len=self.ker_len, n=self.fft_len, con=self.con)
        else:
            x_0 = x
            
        self.gate_outputs = self.gating_network(x_0)

        if not self.training:
            self.gate_outputs = self.gate_outputs / self.moe_temp
        
        noise = torch.randn_like(self.gate_outputs).to(x.device) * self.noise_std
        if self.training:
            noisy_gate_outputs = self.gate_outputs + noise
            self.topk_values, topk_indices = torch.topk(noisy_gate_outputs, self.k, dim=1)
        else:
            self.topk_values, topk_indices = torch.topk(self.gate_outputs, self.k, dim=1)

        self.topk_gates = F.softmax(self.topk_values, dim=1)
        
        batch_size = x.size(0)
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.num_experts)], dim=1)

        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(2))
        sparse_expert_outputs = torch.gather(expert_outputs, 1, topk_indices_expanded)

        output = torch.sum(self.topk_gates.unsqueeze(2) * sparse_expert_outputs, dim=1)

        load_balancing_loss = self.calculate_load_balancing_loss(self.gate_outputs, batch_size)
        
        if get_prob:
            expert_probs = F.softmax(self.gate_outputs, dim=1)
            return output, load_balancing_loss, expert_probs
        
        return output, load_balancing_loss

    def calculate_load_balancing_loss(self, gate_outputs, batch_size):
        gate_probs = F.softmax(gate_outputs, dim=1)
        
        assignments = torch.argmax(gate_outputs, dim=1)
        self.D = torch.zeros(self.num_experts, device=gate_outputs.device)
        for i in range(self.num_experts):
            self.D[i] = torch.sum(assignments == i).float() / batch_size
        
        P = torch.mean(gate_probs, dim=0)
        
        load_balancing_loss = torch.sum(self.D * P) * self.num_experts
        
        return load_balancing_loss


class superLinear(nn.Module):
    def __init__(self, configs):
        super(superLinear, self).__init__()

        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.inf_pred_len = configs.inf_pred_len
        self.max_horizon = configs.max_horizon
        self.auto_regressive = configs.auto_regressive
        self.n_experts = configs.moe_n_experts
        self.moe = configs.moe
        
        if configs.freq_experts == "":
            self.freq_experts = None
        else:
            self.freq_experts = configs.freq_experts.split('_')

        #print("self.freq_experts:", self.freq_experts)

        self.moe_loss = None
        self.top_k_experts = configs.top_k_experts
       # self.noisy_gating = configs.noisy_gating
        self.n_experts = configs.moe_n_experts
        self.freeze_experts = configs.freeze_experts
        self.layer_type = configs.layer_type
        self.model_name = "SuperLinear"

        #print("self.layer_type", self.layer_type)
        self.layer_dict = {'DLinear': DLinear, 'Linear': Linear, 'NLinear': NLinear, 'RLinear': RLinear}
        path = configs.linear_checkpoints_path + configs.linear_checkpoints_dir + "/"
        dirs = os.listdir(path)
        checkpoints_paths = [path + "/" + d + "/" + "checkpoint.pth" for d in dirs]

        if self.freq_experts == "all":
            self.freq_experts = []
            for cp in checkpoints_paths:
                if self.layer_type in cp:
                    cycle = cp.split("/")

        self.experts = {}
        if self.freq_experts is not None:
            for expert_freq in self.freq_experts:
                if expert_freq == "naive" or expert_freq == "Naive":
                    self.experts[expert_freq] = Naive(self.seq_len, self.pred_len)
                elif expert_freq == "mean" or expert_freq == "Mean":    
                    self.experts[expert_freq] = Mean(self.seq_len, self.pred_len)
                else:
                    self.experts[expert_freq] = self.layer_dict[self.layer_type](self.seq_len, self.pred_len)
                    if configs.load_linear:
                        cycle = self.map_to_cycle(expert_freq)
                        cycle_str = f'cycle_{cycle}/'
                        cycle_checkpoint_path = [cp for cp in checkpoints_paths if (cycle_str in cp and self.layer_type in cp)]
                        if len(cycle_checkpoint_path) > 0:
                            print()
                            print(cycle_str)
                            cycle_checkpoint_path = cycle_checkpoint_path[0]
                            #print(f'loading checkpoint with layer type: {self.layer_type} and cycle: {cycle_str}')
                            print(cycle_checkpoint_path)
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
                self.experts[str(i)] = self.layer_dict[self.layer_type](self.seq_len, self.pred_len)

        self.manual_moe = configs.manual_moe

        if configs.misc_moe == 1:
            self.experts["misc"] = self.layer_dict[self.layer_type](self.seq_len, self.pred_len)
            
        self.moe = SparseNoisyMoE(configs, experts=self.experts.values())
        self.dropout = nn.Dropout(configs.dropout)

    def map_to_cycle(self, freq):
        if "/" in freq:
            cycle = int(freq.split("/")[1])
        elif "h" in freq:
            cycle = 24
        elif "2h":
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
        elif "30min" in freq:
            cycle = 48
        else:
            cycle = int(freq)
        return cycle


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, freq=[None], get_prob=False):
        x = x_enc.permute(0, 2, 1)
        B, V, L = x.shape
        x = x.reshape(B * V, L)
        
        expert_probs = None
        
        if get_prob:
            out, self.moe_loss, expert_probs = self.moe(x, get_prob=True)
        else:
            out, self.moe_loss = self.moe(x)

        if self.auto_regressive and self.max_horizon < self.inf_pred_len:
            outputs = [out]
            ar_x = torch.cat([x, out], dim=1)[:, -self.seq_len:]
            for i in range(0, self.inf_pred_len, self.max_horizon):
                ar_out, _ = self.moe(ar_x)
                outputs.append(ar_out)
                ar_x = torch.cat([ar_x, ar_out], dim=1)[:, -self.seq_len:]
            out = torch.cat(outputs, dim=1)[:, :self.inf_pred_len]
        out = out.reshape(B, V, out.shape[-1])
        result = out.permute(0, 2, 1)
        
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
        self.backbone  = superLinear(backbone_cfg)

        # optional final projection: map backbone output to discrete bins
        # (delete if your model already returns logits over a vocabulary)
        self.vocab_size = getattr(config, "vocab_size", None)
        if self.vocab_size is not None:
            self.lm_head = nn.Linear(backbone_cfg.pred_len, self.vocab_size)

        self.post_init()                              # HF weight init

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
        preds = self.backbone(x_enc)[0]
        # if we keep continuous values, treat them as logits directly
        logits = (preds if self.vocab_size is None else self.lm_head(preds).transpose(1, 2))

        loss = None
        if labels is not None:
            # shift for causal objective
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(loss=loss,logits=logits,past_key_values=None,hidden_states=None,attentions=None,)


    def prepare_inputs_for_generation(self, inputs_embeds, past_key_values=None, **kwargs):
        if past_key_values is not None:
            # only feed the last new step
            inputs_embeds = inputs_embeds[:, -1:, :]
        return {"inputs_embeds": inputs_embeds, "past_key_values": past_key_values}

    def _reorder_cache(self, past, beam_idx, **kwargs):
        return past  # backbone keeps no KV cache

"-------------------------------------------------------------------------------------------------------------------"
