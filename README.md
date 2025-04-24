
<div align="center">
  <h2><b>SuperLinear: Foundation Time Series Forecasting</b></h2>
</div>

<div align="center">
  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/azencot-group/SuperLinear?color=green)
![](https://img.shields.io/github/stars/azencot-group/SuperLinear?color=yellow)
![](https://img.shields.io/github/forks/azencot-group/SuperLinear?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>

<div align="center">
  
**[<a href="https://github.com/azencot-group/SuperLinear">GitHub</a>]**
**[<a href="https://your-docs-url.com">Documentation</a>]**

</div>

<p align="center">
  <img src="./figures/logo.png" width="250">
</p>

> Super Linear provides a **comprehensive foundation** for time series forecasting achieveing competitive performance against more complex models
Under efficient architecture combine mixture of frequencies linears experts.



## TODO List
- [ ] Add support for other time series tasks as: probabilistic forecasting, Classifiction, Annomaly ditaction etc'
- [ ] Fine Tuning on specific lookback and horiozn
- [ ] Train from Scratch

## Updates/News:

🚩 **News** (may 2025): Super Linear v1.0.0 has been released!

🚩 **News** (March 2025): 

🚩 **News** (February 2025): 

## Introduction


<p align="center">
  <img src="figures/framework.png" alt="" align="center" width="700px" />
</p>


## 🚀 Getting Started

### Installation

1. Clone the repository
```bash
pip install -r requirements.txt
```

## 📈 Making Forecasts
```typescript
import torch
from transformers import AutoConfig, AutoModelForCausalLM


device                   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch, seq_len, channels = 4, 512, 3  # channels should match your model's expected input channels
series                   = torch.randn(batch, seq_len, channels, dtype=torch.float32).to(device)
torch.manual_seed(42)

# Use the config when loading the model
model = AutoModelForCausalLM.from_pretrained('razmars/SuperLinear',
                                             device_map=device,
                                             torch_dtype='auto',
                                             trust_remote_code=True,
                                             force_download=True)

with torch.no_grad():
    output = model(inputs_embeds=series)
    preds  = output.logits                

```

## Evaluation

+ [Example] Running the follow command to evaluate on ETTh1.

```shell
python run_eval.py -d dataset/ETT-small/ETTh1.csv -p 96
```


## 🔥 Fine-tuning 

⏳ In progress

## 📚 Citation

> Please let us know if you find out a mistake or have any suggestions!

> If you use SuperLinear in your research, please cite:, please consider to star this repository and cite the 
corresponding [paper](https://arxiv.org/pdf/2409.16040):

```
@misc{shi2024timemoe,
      title={Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts}, 
      author={Xiaoming Shi and Shiyu Wang and Yuqi Nie and Dianqi Li and Zhou Ye and Qingsong Wen and Ming Jin},
      year={2024},
      eprint={2409.16040},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2409.16040}, 
}
```
## Related Resources
* Time-Moe: Billion-Scale Time Series Foundation Models with Mixture of Experts, in ICLR 2025. [\[paper\]](https://arxiv.org/abs/2409.16040) [\[GitHub Repo\]](https://github.com/Time-MoE/Time-MoE)
* Foundation Models for Time Series Analysis: A Tutorial and Survey, in *KDD*
  2024. [\[paper\]](https://arxiv.org/abs/2403.14735) [\[Tutorial\]](https://wenhaomin.github.io/FM4TS.github.io/)
* What Can Large Language Models Tell Us about Time Series Analysis, in *ICML*
  2024. [\[paper\]](https://arxiv.org/abs/2402.02713)
* Transformers in Time Series: A Survey, in *IJCAI*
  2023. [\[paper\]](https://arxiv.org/abs/2202.07125) [\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)


## Acknowledgments

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Moe [\[repo\]](ttps://github.com/Time-MoE/Time-MoE)
- Time-LLM  [\[repo\]](https://github.com/KimMeen/Time-LLM)
- TimeMixer [\[repo\]](https://github.com/kwuking/TimeMixer)
- Time-Series-Library [\[repo\]](https://github.com/thuml/Time-Series-Library)
- Large (Language) Models and Foundation Models (LLM, LM, FM) for Time Series and Spatio-Temporal
  Data [\[repo\]](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)

## License

This project is licensed under the MIT License - see the LICENSE file for details.