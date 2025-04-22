from typing import Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    GenerationMixin,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from SuperLinear.model.super_linear_config import SuperLinearConfig

class SuperLinearForCausalLM(PreTrainedModel, GenerationMixin):
    """
    A thin HF wrapper around your MoE backbone.
    * input is expected as `inputs_embeds` – a float tensor shaped (B, L, C)
      where C = number of raw channels (8 for “exchange_rate”, 1 for univariate…)
    * returns CausalLMOutputWithCrossAttentions so it plugs straight into
      transformers’ generation utilities.
    """

    config_class = SuperLinearConfig

    def __init__(self, config: SuperLinearConfig):
        super().__init__(config)
        from SuperLinear.model.sl import Model as Backbone               # your original code :contentReference[oaicite:0]{index=0}

        # the backbone keeps its own Config dataclass, so build one on‑the‑fly:
        backbone_cfg = type("Cfg", (), config.to_dict())()
        self.backbone = Backbone(backbone_cfg)

        # optional final projection: map backbone output to discrete bins
        # (delete if your model already returns logits over a vocabulary)
        self.vocab_size = getattr(config, "vocab_size", None)
        if self.vocab_size is not None:
            self.lm_head = nn.Linear(backbone_cfg.pred_len, self.vocab_size)

        self.post_init()                              # HF weight init

    # ------------------------------------------------------------------
    # Forward pass expected by AutoModelForCausalLM
    # ------------------------------------------------------------------
    def forward(
        self,
        inputs_embeds: torch.Tensor = None,           # (B, L, C) float
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,      # not used by backbone
        use_cache: bool = True,
        labels: Optional[torch.Tensor] = None,        # quantised target ids
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:

        if inputs_embeds is None:
            raise ValueError("Pass the time‑series as `inputs_embeds`")

        # backbone expects (B, C, L)
        x_enc = inputs_embeds.permute(0, 2, 1)
        # backbone returns (B, pred_len, C)
        preds = self.backbone(x_enc)[0]

        # if we keep continuous values, treat them as logits directly
        logits = (
            preds if self.vocab_size is None else self.lm_head(preds).transpose(1, 2)
        )  # (B, L_out, vocab)

        loss = None
        if labels is not None:
            # shift for causal objective
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    # ------------------------------------------------------------------
    # Small helpers needed for `model.generate()`
    # ------------------------------------------------------------------
    def prepare_inputs_for_generation(
        self, inputs_embeds, past_key_values=None, **kwargs
    ):
        if past_key_values is not None:
            # only feed the last new step
            inputs_embeds = inputs_embeds[:, -1:, :]
        return {"inputs_embeds": inputs_embeds, "past_key_values": past_key_values}

    def _reorder_cache(self, past, beam_idx, **kwargs):
        return past  # backbone keeps no KV cache


# 3) --------------------------------------------------------------------------
# REGISTRATION  (one‑liner you run **once** before .from_pretrained)
# -----------------------------------------------------------------------------


AutoConfig.register(SuperLinearConfig.model_type, SuperLinearConfig)
AutoModelForCausalLM.register(SuperLinearConfig, SuperLinearForCausalLM)
