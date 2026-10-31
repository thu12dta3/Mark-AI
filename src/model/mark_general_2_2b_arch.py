from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from .mark_vl_2_2b_arch import MarkVL2_2BConfig, MarkVLForCausalLM, MarkAnyResVisionEncoder, PixelShuffle2DAbstractor, Spatial2DRotaryEmbedding, MarkVLExpertRouter

@dataclass
class MarkGeneral2_2BConfig(MarkVL2_2BConfig):
    model_name: str = 'Mark General 2.2B Multimodal Vision'
    model_version: str = '2.0.0'

class MarkGeneral2_2BForConditionalGeneration(MarkVLForCausalLM):

    def __init__(self, config: Optional[MarkGeneral2_2BConfig]=None):
        if config is None:
            config = MarkGeneral2_2BConfig()
        super().__init__(config)

    def forward(self, input_ids: torch.Tensor, images: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, kv_cache=None, start_pos: int=0, return_mtp: bool=True):
        res = super().forward(input_ids=input_ids, images=images, attention_mask=attention_mask, kv_cache=kv_cache, start_pos=start_pos, return_mtp=return_mtp, labels=labels)
        if isinstance(res, tuple):
            logits_t1, logits_t2 = res
            return {'logits': logits_t1, 'logits_mtp': logits_t2}
        elif isinstance(res, dict):
            return res
        return {'logits': res, 'logits_mtp': None}
