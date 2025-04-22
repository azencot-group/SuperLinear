from typing import Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    GenerationMixin,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# 1) --------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------


class SuperLinearConfig(PretrainedConfig):
    """
    Configuration for the SuperLinear MoE time–series foundation model.
    Only *model_type* must be unique inside transformers; the rest mirrors
    the __init__ arguments of your original Config object.
    """

    model_type = "superlinear-ts"      #  << unique key!

    def __init__(
        self,
        seq_len=512,
        pred_len=96,
        inf_pred_len=96,
        max_horizon=96,
        auto_regressive=False,
        moe_n_experts=8,
        top_k_experts=4,
        moe =1,
        freq_experts='mean_naive_1/6_1/7_1/8_1/12_1/14_1/16_1/21_1/24_1/28_1/30_1/32_1/36_1/42_1/48_1/52_1/56_1/60_1/72_1/84_1/96_1/120_1/144_1/168_1/180_1/224_1/252_1/288_1/336_1/365_1/504_1/672_1/1008_1/1440_1/2016_1/3600',
        **kwargs,                          # any extra CLI args
    ):
        self.seq_len         = seq_len
        self.moe             = moe
        self.pred_len        = pred_len
        self.inf_pred_len    = inf_pred_len
        self.max_horizon     = max_horizon
        self.auto_regressive = auto_regressive
        self.moe_n_experts   = moe_n_experts
        self.top_k_experts   = top_k_experts
        self.freq_experts    = freq_experts
        self.freeze_experts  = 1
        self.layer_type      = "RLinear"
        self.linear_checkpoints_path  = '/cs/azencot_fsas/MoE/'
        self.linear_checkpoints_dir   = "checkpoints5"
        self.load_linear              = 0
        self.manual_moe              = 0
        self.misc_moe                = 1  
        self.noisy_gating_std        = 0.1
        self.noisy_gating_std_decay  = 1
        self.ker_len                 = 50
        self.con                     = 1
        self.d_model                 = 512
        self.mlp_gating              = 1
        self.moe_temp                = 1
        self.use_fft                 = 1
        self.fft_len                 = 10000
        self.dropout                 = 0.0
        super().__init__(**kwargs)
