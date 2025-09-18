
<div align="center">
  <h2><b>Super-Linear: A Lightweight Pretrained Mixture of Linear Experts for Time Series Forecasting</b></h2>
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
  <img src="./figures/logo.png" width="600">
</p>

> Super Linear provides a **comprehensive foundation** for time series forecasting achieveing competitive performance against more complex models
with an efficient architecture combine mixture of frequencies linears experts.

## Updates/News:

🚩 **News** (September 2025): Super Linear v1.0.0 has been released!



## 🚀 Getting Started

### Installation

1. Clone the repository
```bash
pip install -r requirements.txt
```

## 📈 Making Forecasts (Hugging Face)
```typescript
import torch
from transformers import AutoModelForCausalLM

model_path = "./model_files/" # local
model_path = "SequentialLearning/SuperLinear" # hugging_face
model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True, force_download=True)

seq_len = 512
pred_len = 96

freq = 1/32
amp = 1
ph = 0

t = torch.arange(0, seq_len+ pred_len)
s = amp * torch.sin(2 * np.pi * freq * t + ph)
x = s[:seq_len].unsqueeze(0)  # Add batch dim
y = s[seq_len:].unsqueeze(0)  # Add batch dim

with torch.no_grad():
    # takes shapes (B, V, L) or (B, L)
    output = model(x, pred_len=pred_len, get_prob=True)
    preds = output.logits # Predicted values
    probs = output.attentions  # Expert probabilities stored here
expert_names = model.backbone.experts.keys()

```


### Data Preparation

You can obtain all the benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. All the datasets are well pre-processed and can be used easily.

```
mkdir dataset
```
**Please put them in the `./dataset` directory**

## Evaluation

+ [Example] Running the follow command to evaluate on ETTh1 with prediciton horizon 96

```shell
python run_eval.py -d dataset/ETT-small/ETTh1.csv -p 96
```

+ [Example] Running the follow command to evaluate on ETTh1 with prediciton horizon 720

```shell
python run_eval.py -d dataset/ETT-small/ETTh1.csv -p 720
```


+ [Example] Running the follow command to evaluate on weather with prediciton horizon 720

```shell
python run_eval.py -d dataset/weather/weather.csv -p 720
```


## 🔥 Full-Shot, Few-Shot

In progress


## 📚 TODO ⏳

* Add citation 📚
* Add Super-Linear to GIFT-Eval. The results and notebook for GIFT-Eval can be found in the gift-eval directory.

## Related Resources
* Time-Moe: Billion-Scale Time Series Foundation Models with Mixture of Experts, in ICLR 2025. [\[paper\]](https://arxiv.org/abs/2409.16040) [\[GitHub Repo\]](https://github.com/Time-MoE/Time-MoE)
* Foundation Models for Time Series Analysis: A Tutorial and Survey, in *KDD*
  2024. [\[paper\]](https://arxiv.org/abs/2403.14735) [\[Tutorial\]](https://wenhaomin.github.io/FM4TS.github.io/)

## Acknowledgments

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Moe [\[repo\]](https://github.com/Time-MoE/Time-MoE)
- Time-Series-Library [\[repo\]](https://github.com/thuml/Time-Series-Library)


## License

This project is licensed under the MIT License - see the LICENSE file for details.