#!/usr/bin/env python
# run_superlinear_inference.py
# ---------------------------------------------------------------
"""
Unified script to
  (1) optionally convert a legacy checkpoint.pth into a HF checkpoint,
  (2) load the model via AutoModelForCausalLM,
  (3) run one forward pass on a dummy tensor.

Usage examples
--------------
# A) Already have HF folder
python run_superlinear_inference.py --ckpt superlinear-ts

# B) Only have checkpoint.pth
python run_superlinear_inference.py --pth checkpoint.pth --ckpt superlinear-ts
"""
# ---------------------------------------------------------------
import argparse, pathlib, shutil, torch


from SuperLinear.model.modeling_super_linear import SuperLinearForCausalLM           # noqa: F401

from SuperLinear.model.configuration_super_linear   import SuperLinearConfig
from transformers          import AutoConfig, AutoModelForCausalLM


# ---------------------------------------------------------------
def convert_legacy_pth(pth_path: pathlib.Path, ckpt_dir: pathlib.Path):
    """
    Load weights from checkpoint.pth, wrap them in the HF structure,
    and write config.json + pytorch_model.bin to `ckpt_dir`.
    """
    #print(f"[convert]  converting {pth_path}  →  {ckpt_dir}")
    state_dict = torch.load(pth_path, map_location="cpu")
    model      = SuperLinearForCausalLM(SuperLinearConfig())

    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    model.save_pretrained(ckpt_dir)
    #print("[convert]  done.\n")


# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="super_linear",
                        help="HF checkpoint folder (config.json + weights)")
    
    parser.add_argument("--pth",  default="/cs/azencot_fsas/SuperLinear/checkpoints/SuperLinear.pth",
                        help="Optional path to legacy checkpoint.pth to convert")
    
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args     = parser.parse_args()
    ckpt_dir = pathlib.Path(args.ckpt)


    if args.pth is not None:
        convert_legacy_pth(pathlib.Path(args.pth), ckpt_dir)


    device  = torch.device(args.device)
    model   = AutoModelForCausalLM.from_pretrained(ckpt_dir).to(device)
    model.eval()

  
    torch.manual_seed(42)
    batch, seq_len, channels = 4, 512, 3  # channels should match your model's expected input channels
    series = torch.randn(batch, seq_len, channels, dtype=torch.float32).to(device)
    print(f"Input tensor shape: {series.shape}")

    with torch.no_grad():
        output = model(inputs_embeds=series)
        preds  = output.logits                 # (B, pred_len, channels)

    print(preds.shape)


# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
