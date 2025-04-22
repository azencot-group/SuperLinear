
from SuperLinear.layers.Linear_layers import DLinear, Linear, NLinear, RLinear, Naive, Mean

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


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


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

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

        print("self.freq_experts:", self.freq_experts)

        self.moe_loss = None
        self.top_k_experts = configs.top_k_experts
       # self.noisy_gating = configs.noisy_gating
        self.n_experts = configs.moe_n_experts
        self.freeze_experts = configs.freeze_experts
        self.layer_type = configs.layer_type
        self.model_name = "SuperLinear"

        print("self.layer_type", self.layer_type)
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
    