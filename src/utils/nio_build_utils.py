from __future__ import annotations

from typing import Any, Dict

from models.model_NIO import EncoderUSCT, EncoderUSCTHelm2


def extract_nio_build_kwargs(model_init_kwargs: Dict[str, Any] | None) -> Dict[str, int]:
    """Extract numeric kwargs accepted by train_NIO.build_nio."""
    defaults = {
        "usct_hidden": 256,
        "trunk_hidden_layers": 4,
        "trunk_neurons": 128,
        "trunk_n_basis": 256,
        "fno_modes": 16,
        "fno_width": 64,
        "fno_n_layers": 4,
    }
    if not isinstance(model_init_kwargs, dict):
        return defaults

    out = defaults.copy()
    for key in out.keys():
        if key in model_init_kwargs and model_init_kwargs[key] is not None:
            out[key] = int(model_init_kwargs[key])
    return out


def resolve_nio_branch_encoder_cls(model_init_kwargs: Dict[str, Any] | None):
    cls_map = {
        "EncoderUSCT": EncoderUSCT,
        "EncoderUSCTHelm2": EncoderUSCTHelm2,
    }
    if not isinstance(model_init_kwargs, dict):
        return EncoderUSCTHelm2

    raw_value = model_init_kwargs.get("branch_encoder_cls", "EncoderUSCTHelm2")
    if isinstance(raw_value, str):
        return cls_map.get(raw_value, EncoderUSCTHelm2)
    if raw_value in cls_map.values():
        return raw_value
    return EncoderUSCTHelm2


def resolve_nio_branch_encoder_kwargs(model_init_kwargs: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(model_init_kwargs, dict):
        return {}
    kwargs = model_init_kwargs.get("branch_encoder_kwargs", {})
    return kwargs if isinstance(kwargs, dict) else {}
