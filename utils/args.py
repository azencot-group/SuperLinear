import argparse


def get_args(notebook=False):
    parser = argparse.ArgumentParser(description="SuperLinear Model Arguments")
    
    # Model architecture parameters
    parser.add_argument('--train_seq_len', type=int, default=512, help='Training sequence length')
    parser.add_argument('--train_pred_len', type=int, default=96, help='Training prediction length')
    
    # MoE (Mixture of Experts) parameters
    parser.add_argument('--top_k_experts', type=int, default=12, help='Number of top experts to use for inference/training')
    parser.add_argument('--noisy_gating_std', type=float, default=0.1, help='Standard deviation for noisy gating during training')
    parser.add_argument('--moe_temp', type=float, default=1.0, help='Temperature for MoE gating during inference')
    parser.add_argument('--moe_norm', type=bool, default=False, help='Whether to use batch normalization in MoE')
    parser.add_argument('--layer_type', type=str, default='RLinear', help='Type of layer for experts')
    parser.add_argument('--n_experts', type=int, default=4, help='Number of experts when not using freq_experts')
    
    # FFT-based gating parameters
    parser.add_argument('--use_fft', type=bool, default=True, help='Whether to use FFT for gating network')
    parser.add_argument('--fft_len', type=int, default=5000, help='FFT length for periodogram computation')
    
    # Expert configuration
    parser.add_argument('--freq_experts', type=str, default='mean_naive_1/4_1/6_1/7_1/8_1/12_1/14_1/16_1/21_1/24_1/28_1/30_1/32_1/36_1/42_1/48_1/52_1/56_1/60_1/72_1/84_1/90_1/96_1/120_1/144_1/168_1/180_1/224_1/252_1/288_1/336_1/365_1/504_1/672_1/1008_1/1440_1/2016_1/3600', help='Frequency experts separated by underscore (e.g., "1/24_1/48_1/90")')
    parser.add_argument('--comp_moe', type=int, default=12, help='Number of complementary experts to add')
    parser.add_argument('--freeze_experts', type=bool, default=True, help='Whether to freeze expert parameters')
    
    # Model loading and saving
    parser.add_argument('--load_linear', type=bool, default=True, help='Whether to load pre-trained linear experts')
    parser.add_argument('--load_weights_full', type=bool, default=True, help='Whether to load full model weights')
    parser.add_argument('--linear_freq_weights_path', type=str, default='./weights/linear_freq_weights/', help='Path to linear checkpoints')
    parser.add_argument('--full_weights_path', type=str, default='./weights/full_weights/checkpoint.pth', help='Path to model weights')
    
    # Training parameters
    parser.add_argument('--resample_long_lookback', type=bool, default=False, help='Whether to resample long lookback sequences')
    
    if notebook:
        args = parser.parse_args("")
    else:
        args = parser.parse_args()
    return args