#!/usr/bin/env python3
"""
Example usage of SuperLinear model for time series forecasting.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig

def main():
    # Load model configuration and model
    config = AutoConfig.from_pretrained("./", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("./", trust_remote_code=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create sample time series data
    # Shape: [batch_size, sequence_length, features]
    batch_size = 4
    sequence_length = 512
    num_features = 1
    prediction_length = 96
    
    # Generate synthetic time series data
    t = torch.linspace(0, 10, sequence_length)
    sample_data = torch.sin(t).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, num_features)
    
    print(f"Input shape: {sample_data.shape}")
    
    # Generate predictions
    with torch.no_grad():
        outputs = model(inputs_embeds=sample_data, pred_len=prediction_length)
        predictions = outputs.logits
    
    print(f"Prediction shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0, :5, 0]}")  # First 5 predictions of first batch
    
    # Demonstrate with different prediction lengths
    for pred_len in [24, 48, 96, 192]:
        with torch.no_grad():
            outputs = model(inputs_embeds=sample_data, pred_len=pred_len)
            predictions = outputs.logits
        print(f"Prediction length {pred_len}: output shape {predictions.shape}")

if __name__ == "__main__":
    main()
