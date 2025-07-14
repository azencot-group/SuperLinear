from typing import Optional, Tuple

from transformers import PretrainedConfig

# 1) --------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------


class SuperLinearConfig(PretrainedConfig):
    """
    Configuration for the SuperLinear MoE time–series foundation model.
    Only *model_type* must be unique inside transformers; the rest mirrors
    the __init__ arguments of your original Config object.
    """

    model_type = "super_linear"      

    def __init__(
        self,
        # Model architecture parameters
        train_seq_len=512,
        train_pred_len=96,

        # MoE parameters
        top_k_experts=12,
        noisy_gating_std=0.1,
        moe_temp=1.0,
        moe_norm=False,
        layer_type='RLinear',
        comp_moe=12,
        freeze_experts=True,
        
        # FFT-based gating parameters
        use_fft=True,
        fft_len=5000,
        
        # Expert configuration
        freq_experts='mean_naive_1/4_1/6_1/7_1/8_1/12_1/14_1/16_1/21_1/24_1/28_1/30_1/32_1/36_1/42_1/48_1/52_1/56_1/60_1/72_1/84_1/90_1/96_1/120_1/144_1/168_1/180_1/224_1/252_1/288_1/336_1/365_1/504_1/672_1/1008_1/1440_1/2016_1/3600',
        
        # Training parameters
        resample_long_lookback=False,
        
        **kwargs,
    ):
        # Model architecture parameters
        self.train_seq_len = train_seq_len
        self.train_pred_len = train_pred_len
        
        # MoE parameters
        self.top_k_experts = top_k_experts
        self.noisy_gating_std = noisy_gating_std
        self.moe_temp = moe_temp
        self.moe_norm = moe_norm
        self.layer_type = layer_type
        self.comp_moe = comp_moe
        self.freeze_experts = freeze_experts
        
        # FFT-based gating parameters
        self.use_fft = use_fft
        self.fft_len = fft_len
        
        # Expert configuration
        self.freq_experts = freq_experts
        
        # Training parameters
        self.resample_long_lookback = resample_long_lookback
        
        super().__init__(**kwargs)