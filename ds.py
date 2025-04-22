from transformers import AutoModelForCausalLM
import random
from SuperLinear.datasets.time_moe_dataset import TimeMoEDataset

if __name__ == "__main__":


    ds = TimeMoEDataset('Time-300B')
    seq_idx = random.randint(0, len(ds) - 1)
    seq = ds[seq_idx]
    print(f"Sequence {seq_idx}: {seq}")

