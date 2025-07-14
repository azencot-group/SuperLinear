import torch.nn as nn
from layers.RevIN import RevIN


class Naive(nn.Module):
    def __init__(self, input_len, output_len):
        super(Naive, self).__init__()
        self.output_len = output_len

    def forward(self, x):
        # x: [Batch*Channel, Input length]
        x = x[:, -1].unsqueeze(1).repeat(1, self.output_len)
        return x  # to [Batch, Output length, Channel]


class Mean(nn.Module):
    def __init__(self, input_len, output_len):
        super(Mean, self).__init__()
        self.output_len = output_len

    def forward(self, x):
        # x: [Batch*Channel, Input length]
        x = x.mean(dim=1).unsqueeze(1).repeat(1, self.output_len)
        return x  # to [Batch, Output length, Channel]


class RLinear(nn.Module):
    def __init__(self, input_len, output_len):
        super(RLinear, self).__init__()
        self.Linear = nn.Linear(input_len, output_len)
        self.revin_layer = RevIN(num_features=None, affine=False, norm_type=None, subtract_last=False)

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 2:
            x = x.unsqueeze(-1)
        x = x.clone()
        x = self.revin_layer(x, 'norm')
        
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1).clone()
        x = self.revin_layer(x, 'denorm')
        if len(x_shape) == 2:
            x = x.squeeze(-1)
        return x  # to [Batch, Output length, Channel]   
    
