import os
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, Union
import torch
import torch.nn as nn
from src.model.mark_arch import MarkConfig, MarkModel, MarkForCausalLM, PagedKVCache
from src.model.mark_slm_v2_302m_arch import MarkSLMV2Config, MarkSLMV2Model
from src.model.mark_engineer_v2_8b_arch import MarkEngineerV2Config, MarkEngineerV2Model
from src.model.mark_vl_2_2b_arch import MarkVL2_2BConfig, MarkVLForCausalLM
from src.model.mark_v3_arch import MarkV3Config, MarkV3Model, MarkV3ForCausalLM
from src.model.self_rag import SelfRAGController, SpecialTokens
from export.quantize_4bit import NativeIQ4NLLinear, MarkInferenceEngine, load_mark_for_8gb_vram, enforce_vram_ceiling, quantize_mark_model_nf4
quantize_to_iq4_nl = quantize_mark_model_nf4
__version__ = '1.0.0'
__all__ = ['MarkConfig', 'MarkModel', 'MarkForCausalLM', 'MarkSLMV2Config', 'MarkSLMV2Model', 'MarkEngineerV2Config', 'MarkEngineerV2Model', 'MarkVL2_2BConfig', 'MarkVLForCausalLM', 'MarkV3Config', 'MarkV3Model', 'MarkV3ForCausalLM', 'PagedKVCache', 'SelfRAGController', 'SpecialTokens', 'NativeIQ4NLLinear', 'MarkInferenceEngine', 'load_mark_for_8gb_vram', 'enforce_vram_ceiling', 'quantize_to_iq4_nl', 'quantize_mark_model_nf4', 'load_model']

def _has_quantized_weights(sd: Dict[str, Any]) -> bool:
    for k, v in sd.items():
        if k.endswith('.qweight') or (isinstance(v, dict) and 'packed' in v):
            return True
    return False

def load_model(variant: str='302m', checkpoint_path: Optional[str]=None, device: str='auto', max_context_tokens: int=4096) -> Tuple[nn.Module, Any]:
    variant_norm = variant.lower().strip()
    root_dir = Path(__file__).resolve().parent.parent
    if checkpoint_path is None:
        candidate_names = [f'mark_{variant_norm}.pt', f'mark_engineer_{variant_norm}.pt', f'mark_{variant_norm}_iq4_nl.pt', f'mark_engineer_{variant_norm}_iq4_nl.pt', f'mark_{variant_norm}_inference.pt', f'mark_engineer_{variant_norm}_inference.pt', f'mark_{variant_norm}_final.pt', f"mark_{variant_norm.replace('.', '_')}_iq4_nl.pt", f"mark_engineer_{variant_norm.replace('.', '_')}_iq4_nl.pt", f"mark_general_{variant_norm.replace('.', '_')}_iq4_nl.pt"]
        if variant_norm in ('302m', '300m', 'engineer_300m'):
            candidate_names.extend(['mark_engineer_302m_iq4_nl.pt', 'mark_engineer_300m_iq4_nl.pt'])
        elif variant_norm in ('1.5b', '1_5b', 'engineer-1.5b'):
            candidate_names.extend(['mark_engineer_1_5b_iq4_nl.pt', 'mark_engineer_1.5b_iq4_nl.pt'])
        elif variant_norm in ('general', 'general-1.5b', 'general_1.5b'):
            candidate_names.extend(['mark_general_1_5b_iq4_nl.pt', 'mark_general_1.5b_iq4_nl.pt'])
        elif variant_norm in ('v2-8b', 'v2_8b'):
            candidate_names.extend(['mark_engineer_v2_8b_step_15000.pt', 'mark_engineer_v2_8b_iq4_nl.pt'])
        elif variant_norm in ('v3-1.1b', 'v3', 'v3_1.1b'):
            candidate_names.extend(['mark_v3_core_1_1b.pt', 'mark_v3_1_1b_iq4_nl.pt'])
        elif variant_norm in ('slm-v3-150m', 'mark-slm-v3-150m', 'slm_v3_150m'):
            candidate_names.extend(['mark_slm_v3_150m.pt', 'mark_slm_v3_150m_iq4_nl.pt'])
        elif variant_norm in ('general-v3-1.5b', 'mark-general-v3-1.5b', 'general_v3_1_5b', 'general-v3-1_5b'):
            candidate_names.extend(['mark_general_v3_1_5b.pt', 'mark_general_v3_1_5b_iq4_nl.pt', 'mark_general_v3_1.5b.pt'])
        elif variant_norm in ('engineer-v3-1.5b', 'mark-engineer-v3-1.5b', 'engineer_v3_1_5b', 'engineer-v3-1_5b'):
            candidate_names.extend(['mark_engineer_v3_1_5b.pt', 'mark_engineer_v3_1_5b_iq4_nl.pt', 'mark_engineer_v3_1.5b.pt'])
        elif variant_norm in ('experimental-v3-8b', 'mark-experimental-v3-8b', 'experimental_v3_8b'):
            candidate_names.extend(['mark_experimental_v3_8b.pt', 'mark_experimental_v3_8b_iq4_nl.pt'])
        elif variant_norm in ('markvl-2.2b', 'markvl', 'markvl_2_2b'):
            candidate_names.extend(['mark_vl_2_2b.pt', 'mark_vl_2_2b_iq4_nl.pt'])
        search_dirs = [root_dir / 'checkpoints', Path('checkpoints'), Path('/mnt/ssd_1/mark ai/checkpoints'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v2_8b')]
        for d in search_dirs:
            for name in candidate_names:
                p = d / name
                if p.exists():
                    checkpoint_path = str(p)
                    break
            if checkpoint_path:
                break
    sd = None
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    if variant_norm in ('302m', '300m', 'engineer_300m'):
        config = MarkConfig.create_engineer_300m()
        model_cls = MarkModel
    elif variant_norm in ('slm-v2-302m', 'slm_v2_302m', 'slm_v2'):
        config = MarkSLMV2Config()
        model_cls = MarkSLMV2Model
    elif variant_norm in ('1.5b', '1_5b', 'engineer-1.5b', 'engineer_1_5b'):
        config = MarkConfig.create_engineer_1_5b()
        model_cls = MarkModel
    elif variant_norm in ('general', 'general-1.5b', 'general_1.5b'):
        config = MarkConfig.create_general_1_5b()
        model_cls = MarkModel
    elif variant_norm in ('markvl-2.2b', 'markvl', 'markvl_2_2b', '2.2b'):
        config = MarkVL2_2BConfig()
        model_cls = MarkVLForCausalLM
    elif variant_norm in ('v2-8b', 'v2_8b', 'engineer-v2-8b'):
        config = MarkEngineerV2Config()
        model_cls = MarkEngineerV2Model
    elif variant_norm in ('slm-v3-150m', 'mark-slm-v3-150m', 'slm_v3_150m'):
        config = MarkV3Config.create_slm_v3_150m()
        model_cls = MarkV3ForCausalLM
    elif variant_norm in ('general-v3-1.5b', 'mark-general-v3-1.5b', 'general_v3_1_5b', 'general-v3-1_5b'):
        config = MarkV3Config.create_general_v3_1_5b()
        model_cls = MarkV3ForCausalLM
    elif variant_norm in ('engineer-v3-1.5b', 'mark-engineer-v3-1.5b', 'engineer_v3_1_5b', 'engineer-v3-1_5b'):
        config = MarkV3Config.create_engineer_v3_1_5b()
        model_cls = MarkV3ForCausalLM
    elif variant_norm in ('experimental-v3-8b', 'mark-experimental-v3-8b', 'experimental_v3_8b'):
        config = MarkV3Config.create_experimental_v3_8b()
        model_cls = MarkV3ForCausalLM
    elif variant_norm in ('v3-1.1b', 'v3', 'v3_1.1b', 'core-1.1b'):
        config = MarkV3Config()
        model_cls = MarkV3ForCausalLM
    elif variant_norm in ('primal', 'base', '0.1'):
        config = MarkConfig.create_primal()
        model_cls = MarkModel
    else:
        raise ValueError(f"Unknown variant: '{variant}'.")
    if sd is not None:
        emb_weight = sd.get('tok_embeddings.weight') or sd.get('model.embed_tokens.weight') or sd.get('embed_tokens.weight')
        if emb_weight is not None and hasattr(emb_weight, 'shape'):
            if hasattr(config, 'vocab_size'):
                config.vocab_size = emb_weight.shape[0]
    model = model_cls(config)
    if sd is not None:
        is_quantized = _has_quantized_weights(sd)
        if is_quantized:
            quantize_mark_model_nf4(model, weight_quant_type='IQ4_NL')
            model.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(sd, strict=False)
        print(f'[Mark AI] Weights loaded successfully from: {checkpoint_path}')
    else:
        print(f"[Mark AI] Initialized with structural weights for variant '{variant_norm}'.")
    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)
    model = model.to(dev)
    return (model, config)
