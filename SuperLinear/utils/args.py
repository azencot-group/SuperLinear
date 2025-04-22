# %%
import argparse
import os
import torch
import copy
import time

from create_run_script import *
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
#import neptune

def get_args(notebook=False):
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data_type', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='batch size')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # PatchTST
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # added by liran
    parser.add_argument('--seed', type=int, default=2021, help='seed')
    parser.add_argument('--exp', type=str, default="exp0", help='experiment tag')
    parser.add_argument('--method', type=str, default='none', help='method type [none,synth, synth_minimix]')
    parser.add_argument('--scale_type', type=str, default='standard', help='[standard, minmax, maxabs, robust, power, none]')
    parser.add_argument('--model_scaler', type=str, default='none', help='[standard, minmax, maxabs, robust, power, none]')
    parser.add_argument('--override_table', type=int, default=0, help='override table')   
    parser.add_argument('--save_predictions', type=int, default=0, help='')
    parser.add_argument('--configs_path', type=str, default='./monitor/configs/', help='root path of the configs file')
    parser.add_argument('--use_configs', type=int, default=1, help='whether to use configs')
    parser.add_argument('--all_configs', type=int, default=0, help='whether to run all the configs table')
    parser.add_argument('--inverse_metrics', type=int, default=0, help='')
    parser.add_argument('--delete_weights', type=int, default=1, help='seed')
    parser.add_argument('--freq_enc', type=int, default=0, help='')
    parser.add_argument('--instruct', type=str, default="", help='')
    parser.add_argument('--reconstruct', type=int, default=0, help='')
    parser.add_argument('--inf_pred_len', type=str, default=-1, help='-1 for normal pred_len')
    parser.add_argument('--resample_freq', type=str, default="", help='')
    parser.add_argument('--nlinear_norm', type=int, default=0, help='')
    # cycle
    parser.add_argument('--cycle', type=int, default=0, help='')

    # grad manip
    parser.add_argument('--grad_inference', type=int, default=0, help='')
    parser.add_argument('--layerwise', type=int, default=0, help='')
    parser.add_argument('--mag_sim', type=int, default=0, help='')
    parser.add_argument('--task_type', type=str, default="instruct", help='')
    parser.add_argument('--grad_inference_path', type=str, default='/cs/azencot_fsas/Synth/grad_inference/', help='')
    #parser.add_argument('--grad_manip_layers', type=str, default="", help='')

    # linear probing
    parser.add_argument('--linear_probing', type=int, default=0, help='')
    parser.add_argument('--inf_linear_probing', type=int, default=0, help='')
    parser.add_argument('--linear_probing_epochs', type=int, default=10, help='')
    parser.add_argument('--linear_probing_lr', type=float, default=0.001, help='')
    parser.add_argument('--linear_probing_patience', type=int, default=3, help='')
    parser.add_argument('--linear_probing_lradj', type=str, default='type1', help='')

    # neptune
    parser.add_argument('--neptune', type=int, default=0, help='whether to use neptune')
    parser.add_argument('--project', type=str, default="azencot-group/Ablation-MultiTask", help='neptune project name')
    parser.add_argument('--api_token', type=str, default="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZDU5ZjRiOC04NjljLTQ1NGYtOTFjMi01ODRjZjBjNWE1MjgifQ==", help='neptune project name')

    parser.add_argument("--synth", type=int, default=0, help="use synth or not")


     # GPT4TS
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--cos', type=int, default=0)
    parser.add_argument('--decay_fac', type=float, default=0.75)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--tmax', type=int, default=10)

    # TTM
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--decoder_num_layers', type=int, default=2)
    parser.add_argument('--adaptive_patching_levels', type=int, default=0)
    parser.add_argument('--frequency_token_vocab_size', type=int, default=11)
    
    # TTM_ft
    parser.add_argument('--freeze_backbone', type=int, default=1)
    # TimeLLM
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') 


    #TimeMixer
    parser.add_argument('--use_future_temporal_feature', type=int, default=0, help='LLM model')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',help='down sampling method, only support avg, max, conv')
    # Timer    #Units

    parser.add_argument('--num_samples', type=int, default=100, help='') 
    parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint path for timer') 
    

    #Moment
    parser.add_argument('--orth_gain', type=float, default=1.41)
    parser.add_argument('--mask_ratio', type=float, default=1.41)
    parser.add_argument('--randomly_initialize_backbone', type=int, default=0)
    parser.add_argument('--transformer_type', type=str, default='encoder_decoder')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--eta_min', type=float, default=1e-8)

    # moirai_ft
    parser.add_argument('--moirai_patch_len', type=str, default="auto")

    # UniTime
    parser.add_argument('--dec_head_dropout', type=float, default=0.1, help='decoder head dropout')
    parser.add_argument('--dec_trans_layer_num', type=int, default=2, help='decoder transformer layer number')
    parser.add_argument('--max_token_num', type=int, default=17, help='maximum token number')
    parser.add_argument('--pre_model', type=str, default="gpt2", help='pretrained language model from hugging face [gpt2]')
    parser.add_argument('--clip', type=int, default=0, help='gradient clipping')
    parser.add_argument('--lm_ft_type', type=str, default='full', help='')
  
    parser.add_argument("--ker_len", type=int, default=25, help="")
    parser.add_argument("--harmonics", type=int, default=3, help="")
    parser.add_argument("--base_freq", type=float, default=1/24, help="")
    parser.add_argument("--avg_amp", type=float, default=5, help="")
    parser.add_argument("--amp_dist", type=str, default="exponential", help="")

    # synth
    parser.add_argument("--pool_f", type=float, default=0.1 , help="the frac of the pool size to sample from")
    parser.add_argument("--pool_size", type=int, default=100 , help="number of pool size")
    parser.add_argument("--synth_length", type=int, default=1024, help="number of timesteps")
    parser.add_argument("--size_like", type=str, default='' , help="size like other dataset")
    parser.add_argument("--output_name", type=str, default="output.csv")
    parser.add_argument("--output_path", type=str, default="dataset/synth/")

    parser.add_argument("--freq_skip", type=float, default=0.01, help="for gauss")
    parser.add_argument("--max_freq", type=float, default=0.2, help="for gauss")
    parser.add_argument("--min_freq", type=float, default=0.001, help="for gauss")
    parser.add_argument("--sample_dataset", type=str, default='', help="electricity,weather...")
    parser.add_argument("--k_best", type=int, default=5, help="for gauss")
    parser.add_argument("--minimix", type=int, default=3,help="the number mof different synthetic datasets")
    parser.add_argument("--use_gauss", type=int, default=0)
    parser.add_argument("--rect", type=int, default=0)
    parser.add_argument("--dataset_maxsize", type=int, default=np.inf)
    parser.add_argument("--total_size", type=int, default=np.inf)
    parser.add_argument("--linear_identity", type=int, default=0)
    parser.add_argument("--multi", type=int, default=0)
    parser.add_argument("--exclude_hourly", type=int, default=0)
    parser.add_argument("--exclude_daily", type=int, default=0)
    parser.add_argument("--rw_pool_size", type=int, default=0)
    parser.add_argument("--rw_std", type=int, default=1)
    parser.add_argument("--arp_pool_size", type=int, default=100)
    parser.add_argument("--arp_type", type=str, default="avg")
    parser.add_argument("--arp_k", type=int, default=8)
    parser.add_argument("--pw_pool_size", type=int, default=0)
    parser.add_argument("--use_rw", type=int, default=0)
    parser.add_argument("--use_pw", type=int, default=0)

    #CycleNet
    parser.add_argument("--use_revin", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="linear")
    


    # MoE
    parser.add_argument("--moe", type=int, default=0)
    parser.add_argument("--moe_n_experts", type=int, default=20)
    parser.add_argument("--moe_lambda", type=int, default=1)
    parser.add_argument("--top_k_experts", type=int, default=4)
    parser.add_argument("--noisy_gating", type=int, default=1)
    parser.add_argument("--noisy_gating_std", type=int, default=1)
    parser.add_argument("--moe_hidden_size", type=int, default=100)
    parser.add_argument("--moe_linear_experts", type=int, default=100)
    parser.add_argument("--moe_mlp_experts", type=int, default=100)
    parser.add_argument("--moe_nonlinear_experts", type=int, default=100)
    parser.add_argument("--moe_activation", type=str, default='relu',help="relu,tanh")
    

    # moe linear
    parser.add_argument("--moe_temp", type=int, default=0.01)
    parser.add_argument("--freeze_experts", type=int, default=1)
    parser.add_argument("--load_linear", type=int, default=1)
    parser.add_argument("--freq_experts", type=str, default='',help="relu,tanh")
    parser.add_argument("--fft_len", type=int, default=10000, help='Linear, DLinear, NLinear, RLinear')
    parser.add_argument("--con", type=int, default=1, help='Linear, DLinear, NLinear, RLinear')
    parser.add_argument("--use_fft", type=int, default=1, help='Linear, DLinear, NLinear, RLinear')
    # max_horizon
    parser.add_argument("--max_horizon", type=int, default=96, help='Linear, DLinear, NLinear, RLinear')
    #autoregressive
    parser.add_argument("--auto_regressive", type=int, default=0, help='Linear, DLinear, NLinear, RLinear')
    # max_resample_factor
    parser.add_argument("--max_resample_factor", type=int, default=4, help='')


    #args.manual_moe = 1
    parser.add_argument("--dual_stage", type=int, default=0, help='')
    parser.add_argument("--dual_stage_epochs", type=int, default=30, help='')
    parser.add_argument("--misc_moe", type=int, default=1, help='')
    parser.add_argument("--moe_patience", type=int, default=20, help='')
    parser.add_argument("--moe_learning_rate", type=float, default=0.1, help='')
    parser.add_argument("--manual_moe", type=int, default=0, help='')



    # layer_type
    parser.add_argument("--layer_type", type=str, default='Linear', help='Linear, DLinear, NLinear, RLinear')

    
    # transfer learning
    parser.add_argument("--synth_scaler", type=str, default="standard", help="1 for random, 2 for seasonal")
    parser.add_argument("--transfer_method_list", type=str, default="zeroshot_fewshot5")
    parser.add_argument("--transfer_method", type=str, default="")
    parser.add_argument("--is_inference", type=int, default=0)
    parser.add_argument("--inference_batch_size", type=int, default=512)
    parser.add_argument("--fewshot_train_epochs", type=int, default=5)
    parser.add_argument("--synth_config", type=int, default=0, help="1 for random, 2 for seasonal")
    parser.add_argument("--minimix_max_freqs", type=str, default="", help="l")
    parser.add_argument("--minimix_freqs", type=str, default="", help="l")
    parser.add_argument("--minimix_harmonics", type=str, default=("3_2_1_"*10)[:-1], help="l")
    parser.add_argument("--minimix_amps", type=str, default="5", help="l")


    parser.add_argument("--full_size", type=int, default=0, help="l")
    parser.add_argument("--data_list", type=str, default="", help="l")
    parser.add_argument("--inf_pred_len_list", type=str, default="", help="l")
    parser.add_argument("--inference_data_list", type=str, default='Births_Sunspot_Saugeen_Solar_weather_ETTm2_ETTm1_electricity_ETTh1_ETTh2_traffic_Exchange_KDD_Wind_Demand_Bitcoin_PEMS03', help="l")
    # channel ind
    parser.add_argument("--channel_ind", type=int, default=0, help="l")

    if notebook:
        args = parser.parse_args("")
        args.use_configs = 0
        args.override_table = 1
    else:
        args = parser.parse_args()
    return args

# get run configuration table
def get_configs(args):
    key_index_names = []
    key_index_values = []

    # if a table exists, get the table
    if args.use_configs:
        configs = pd.read_csv(args.configs_path + args.exp + '.csv')
        if not args.all_configs:
            # set the first 2 columns index
            configs_key_index = np.arange(int(configs['maxi_index'][0]))
            key_index_names =  list(configs.columns[configs_key_index])
            # args to dict
            key_index_values = [vars(args)[key] for key in key_index_names]
            configs = configs[(configs[key_index_names] == key_index_values).all(axis=1)]
        configs.drop(columns=['maxi_index'], inplace = True)

    # create arbitrary table
    else:
        data = {'exp': [args.exp],
                'model_id': [args.model_id],
                'model': [args.model],
                'seq_len': [args.seq_len],
                'pred_len': [args.pred_len],
                'seed': [args.seed]}
        configs  = pd.DataFrame(data)



    # create table monitor
    folder_path = os.getcwd()+'/monitor/table_monitor_sandbox/'
    setting = args.exp
    for v in key_index_values:
        setting = setting + '_' + str(v)

    table_id_name = setting
    table_path = folder_path+table_id_name+'.csv'

    # check of table exists
    if os.path.isfile(table_path) and not args.override_table:
        print("openinig experiment: ", table_id_name)
        run_configs = pd.read_csv(table_path)

    else: 
        if args.override_table:
            print("overriding experiment: ", table_id_name)
        else:
            print("creating experiment: ", table_id_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path) 
        run_configs = configs.copy()
    # run_configs[['mae', 'mse', 'rmse', 'mape', 'mspe']] = None 
    # run_configs[['inv_mae', 'inv_mse', 'inv_rmse', 'inv_mape', 'inv_mspe']] = None 
        run_configs['complete'] = None
        run_configs['inference_complete'] = None
        run_configs.to_csv(table_path,index=False)
    return run_configs