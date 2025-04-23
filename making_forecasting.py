import torch
from transformers import AutoConfig, AutoModelForCausalLM



# Use the config when loading the model
model = AutoModelForCausalLM.from_pretrained(
    'razmars/SuperLinear',
    device_map="cuda",
    trust_remote_code=True,
)
torch.manual_seed(42)
device                   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch, seq_len, channels = 4, 512, 3  # channels should match your model's expected input channels
series                   = torch.randn(batch, seq_len, channels, dtype=torch.float32).to(device)


with torch.no_grad():
    output = model(inputs_embeds=series)
    preds  = output.logits                
print(preds.shape)