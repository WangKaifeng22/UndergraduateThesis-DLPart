from typing import Any, Dict, Optional

from model_Unet_CNN import FourierDeepONet
from model_original import FourierDeepONet_Origin


_FOURIER_ORIGINAL_ALLOWED_KEYS = {
    "num_parameter",
    "width",
    "modes1",
    "modes2",
    "regularization",
    "merge_operation",
}


def is_original_fourier_deeponet_config(model_config: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(model_config, dict):
        return False
    return bool(model_config.get("is_original", False))


def build_fourier_deeponet_variant(
    model_init_kwargs: Optional[Dict[str, Any]] = None,
    trunk_dim: Optional[int] = None,
    original: bool = False,
    width: int = 64,
    modes1: int = 12,
    modes2: int = 20,
    regularization=None,
    merge_operation: str = "mul",
    use_hfs_block123: bool = True,
    hfs_patch_size=(16, 8, 4),
):
    if regularization is None:
        regularization = ["l2", 3e-6]

    kwargs = dict(model_init_kwargs or {})
    if original:
        kwargs = {k: v for k, v in kwargs.items() if k in _FOURIER_ORIGINAL_ALLOWED_KEYS}

    if trunk_dim is not None:
        kwargs.setdefault("num_parameter", trunk_dim)
    kwargs.setdefault("width", width)
    kwargs.setdefault("modes1", modes1)
    kwargs.setdefault("modes2", modes2)
    kwargs.setdefault("regularization", regularization)
    kwargs.setdefault("merge_operation", merge_operation)

    if original:
        return FourierDeepONet_Origin(**kwargs)

    kwargs.setdefault("use_hfs_block123", use_hfs_block123)
    kwargs.setdefault("hfs_patch_size", hfs_patch_size)
    return FourierDeepONet(**kwargs)
