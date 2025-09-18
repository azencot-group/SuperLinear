
---
license: mit
tags:
  - time-series
  - mixture-of-experts
  - forecasting
  - pytorch
  - fft
model-index:
  - name: SuperLinear
    results: []
---


# SuperLinear: A Mixture of Experts Time Series Forecasting Model

SuperLinear is a novel time series forecasting model that employs a Mixture of Experts (MoE) architecture to achieve superior performance across various forecasting tasks. The model routes inputs to the most relevant experts based on frequency-domain analysis using FFT-based gating networks.

## Model Architecture

The SuperLinear model consists of:

- **Sparse Mixture of Experts (MoE)**: Routes inputs to the top-k most relevant experts
- **FFT-based Gating Network**: Uses frequency domain analysis to determine expert routing
- **Frequency-specific Experts**: Pre-trained experts specialized for different temporal patterns

## Key Features

- **Adaptive Expert Selection**: Dynamic routing based on input characteristics
- **Frequency-aware Processing**: Leverages FFT analysis for intelligent expert selection
- **Auto-regressive Capabilities**: Supports long-horizon forecasting
- **Multi-scale Processing**: Handles various sequence lengths through resampling

## Usage

```python
from transformers import AutoModelForCausalLM, AutoConfig
import torch

# Load the model
model = AutoModelForCausalLM.from_pretrained("SequentialLearning/SuperLinear", trust_remote_code=True)

# Prepare input time series data
# Shape: [batch_size, sequence_length, features]
input_data = torch.randn(1, 512, 1)

# Generate predictions
with torch.no_grad():
    outputs = model(inputs_embeds=input_data, pred_len=96)
    predictions = outputs.logits  # Shape: [batch_size, prediction_length, features]
```

## Configuration

Key configuration parameters:

- `train_seq_len`: Training sequence length (default: 512)
- `train_pred_len`: Training prediction length (default: 96)
- `top_k_experts`: Number of experts to use (default: 12)
- `use_fft`: Whether to use FFT-based gating (default: True)
- `freq_experts`: Frequency-specific expert configuration
- `moe_temp`: Temperature for expert selection during inference (default: 1)

## Link to GitHub

[https://github.com/azencot-group/SuperLinear](https://github.com/azencot-group/SuperLinear)

## Citation

If you use SuperLinear in your research, please cite:

```bibtex
@article{todo,
  title={SuperLinear: todo},
  author={Your Name},
  year={2025}
}
```

## License

This model is released under the MIT License.
