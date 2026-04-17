import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from InversionNet import InversionNet
from model_Unet_CNN import FourierDeepONet
from nio_build_utils import (
    extract_nio_build_kwargs,
    resolve_nio_branch_encoder_cls,
    resolve_nio_branch_encoder_kwargs,
)
from train_NIO import build_nio


MODEL_TYPE_MAP = {
    "fourierdeeponet": "fourier_deeponet",
    "fourier_deeponet": "fourier_deeponet",
    "nioultrasoundctabl": "nio",
    "nio": "nio",
    "inversionnet": "inversionnet",
}


def make_config_tag(config_path: Path, index: int) -> str:
    """Generate a stable, filesystem-safe tag for config-specific outputs."""
    safe_stem = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in config_path.stem)
    return f"cfg_{index:02d}_{safe_stem}"


def _shape_to_str(shape: torch.Size) -> str:
    return "x".join(str(v) for v in shape)


def count_model_parameters(model: torch.nn.Module) -> Tuple[List[Dict], List[Dict], Dict]:
    """Return parameter-level rows, layer-level rows, and model totals."""
    param_rows: List[Dict] = []
    layer_counts: Dict[str, int] = defaultdict(int)

    total_params = 0
    total_trainable = 0

    for param_name, param in model.named_parameters():
        numel = int(param.numel())
        is_trainable = bool(param.requires_grad)
        total_params += numel
        if is_trainable:
            total_trainable += numel

        layer_name = param_name.rsplit(".", 1)[0] if "." in param_name else param_name
        if is_trainable:
            layer_counts[layer_name] += numel

        param_rows.append(
            {
                "param_name": param_name,
                "layer_name": layer_name,
                "shape": _shape_to_str(param.shape),
                "numel": numel,
                "trainable": is_trainable,
            }
        )

    layer_rows = [
        {"layer_name": layer_name, "trainable_params": count}
        for layer_name, count in sorted(layer_counts.items(), key=lambda x: x[0])
    ]

    totals = {
        "total_params": total_params,
        "total_trainable_params": total_trainable,
        "total_non_trainable_params": total_params - total_trainable,
    }

    return param_rows, layer_rows, totals


def build_fourier_deeponet(
    num_parameter: int = 2,
    width: int = 64,
    modes1: int = 12,
    modes2: int = 20,
    regularization: Optional[List[Any]] = None,
    merge_operation: str = "mul",
    use_hfs_block123 = True,
    hfs_patch_size = (16, 8, 4),
) -> torch.nn.Module:
    if regularization is None:
        regularization = ["l2", 3e-6]
    return FourierDeepONet(
        num_parameter=num_parameter,
        width=width,
        modes1=modes1,
        modes2=modes2,
        regularization=regularization,
        merge_operation=merge_operation,
        use_hfs_block123=use_hfs_block123,
        hfs_patch_size=hfs_patch_size,
    )


def build_inversionnet(
    dim0: int = 64,
    dim1: int = 64,
    dim2: int = 64,
    dim3: int = 128,
    dim4: int = 256,
    dim5: int = 512,
    regularization: Optional[List[Any]] = None,
) -> torch.nn.Module:
    if regularization is None:
        regularization = ["l2", 3e-6]
    return InversionNet(
        dim0=dim0,
        dim1=dim1,
        dim2=dim2,
        dim3=dim3,
        dim4=dim4,
        dim5=dim5,
        regularization=regularization,
    )


def normalize_model_key(model_type: str) -> str:
    key = model_type.strip().lower()
    if key not in MODEL_TYPE_MAP:
        raise ValueError(f"Unsupported model_type in config: {model_type}")
    return MODEL_TYPE_MAP[key]


def load_model_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("model_config.json must be a JSON object.")
    return payload


def get_model_kwargs_from_config(model_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "model_type" in cfg:
        cfg_model_key = normalize_model_key(str(cfg["model_type"]))
        if cfg_model_key != model_key:
            return {}
        kwargs = cfg.get("model_init_kwargs", {})
        return kwargs if isinstance(kwargs, dict) else {}

    models_block = cfg.get("models")
    if isinstance(models_block, dict) and model_key in models_block:
        entry = models_block[model_key]
        if isinstance(entry, dict):
            if "model_init_kwargs" in entry and isinstance(entry["model_init_kwargs"], dict):
                return entry["model_init_kwargs"]
            return entry

    direct_entry = cfg.get(model_key)
    if isinstance(direct_entry, dict):
        if "model_init_kwargs" in direct_entry and isinstance(direct_entry["model_init_kwargs"], dict):
            return direct_entry["model_init_kwargs"]
        return direct_entry

    return {}


def save_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count trainable parameters for Fourier DeepONet, NIO, InversionNet.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["fourier_deeponet", "nio", "inversionnet"],
        choices=["fourier_deeponet", "nio", "inversionnet"],
        help="Models to count.",
    )
    parser.add_argument("--output-dir", type=str, default="reports/param_counts", help="Output directory.")
    parser.add_argument("--fourier-num-parameter", type=int, default=2, help="Trunk input dim for FourierDeepONet.")
    parser.add_argument("--nio-usct-time-steps", type=int, default=1900, help="USCT time steps for NIO encoder.")
    parser.add_argument("--nio-usct-hidden", type=int, default=256, help="Hidden size for NIO encoder.")
    parser.add_argument("--nio-trunk-hidden-layers", type=int, default=4, help="Hidden layers in NIO trunk MLP.")
    parser.add_argument("--nio-trunk-neurons", type=int, default=128, help="Hidden width in NIO trunk MLP.")
    parser.add_argument("--nio-trunk-n-basis", type=int, default=256, help="Output basis size of NIO trunk/branch.")
    parser.add_argument("--nio-fno-modes", type=int, default=20, help="Fourier modes used in NIO FNO.")
    parser.add_argument("--nio-fno-width", type=int, default=64, help="Channel width used in NIO FNO.")
    parser.add_argument("--nio-fno-n-layers", type=int, default=4, help="Number of layers in NIO FNO.")
    parser.add_argument(
        "--nio-trunk-activation",
        type=str,
        default="relu",
        help="Deprecated and ignored when using train_NIO.build_nio (kept for CLI compatibility).",
    )
    parser.add_argument("--seed", type=int, default=114514, help="Seed used in NIO config.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for NIO build.")
    parser.add_argument(
        "--model-config",
        nargs="+",
        default=None,
        help="One or more model_config.json paths. If set, model constructor params are read from each config and manual CLI model params are ignored.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_totals: List[Dict] = []

    run_items: List[Tuple[str, Dict[str, Any], Optional[str], List[str]]] = []
    if args.model_config:
        for idx, cfg_raw_path in enumerate(args.model_config, start=1):
            cfg_path = Path(cfg_raw_path)
            if not cfg_path.exists():
                raise FileNotFoundError(f"model_config not found: {cfg_path}")
            cfg_payload = load_model_config(cfg_path)
            models_to_run = [normalize_model_key(str(cfg_payload["model_type"]))] if "model_type" in cfg_payload else list(args.models)
            config_tag = make_config_tag(cfg_path, idx)
            run_items.append((config_tag, cfg_payload, str(cfg_path.resolve()), models_to_run))
            print(f"Using model config [{config_tag}]: {cfg_path.resolve()}")
    else:
        run_items.append(("cli", {}, None, list(args.models)))

    for config_tag, model_cfg, config_source, models_to_run in run_items:
        config_output_dir = output_dir / config_tag
        config_output_dir.mkdir(parents=True, exist_ok=True)

        for model_key in models_to_run:
            cfg_kwargs = get_model_kwargs_from_config(model_key, model_cfg) if model_cfg else {}

            if model_key == "fourier_deeponet":
                model = build_fourier_deeponet(
                    num_parameter=int(cfg_kwargs.get("num_parameter", args.fourier_num_parameter)),
                    width=int(cfg_kwargs.get("width", 64)),
                    modes1=int(cfg_kwargs.get("modes1", 12)),
                    modes2=int(cfg_kwargs.get("modes2", 20)),
                    regularization=cfg_kwargs.get("regularization", ["l2", 3e-6]),
                    merge_operation=str(cfg_kwargs.get("merge_operation", "mul")),
                )
            elif model_key == "nio":
                requested_trunk_activation = str(cfg_kwargs.get("trunk_activation", args.nio_trunk_activation))
                if "trunk_activation" in cfg_kwargs or args.nio_trunk_activation != "relu":
                    print(
                        f"[warn] trunk_activation='{requested_trunk_activation}' is ignored "
                        "because count_trainable_params now reuses train_NIO.build_nio."
                    )

                nio_kwargs = extract_nio_build_kwargs(cfg_kwargs)
                branch_encoder_cls = resolve_nio_branch_encoder_cls(cfg_kwargs)
                branch_encoder_kwargs = resolve_nio_branch_encoder_kwargs(cfg_kwargs)

                model = build_nio(
                    seed=args.seed,
                    usct_time_steps=int(cfg_kwargs.get("usct_time_steps", args.nio_usct_time_steps)),
                    device=device,
                    usct_hidden=int(nio_kwargs.get("usct_hidden", args.nio_usct_hidden)),
                    trunk_hidden_layers=int(nio_kwargs.get("trunk_hidden_layers", args.nio_trunk_hidden_layers)),
                    trunk_neurons=int(nio_kwargs.get("trunk_neurons", args.nio_trunk_neurons)),
                    trunk_n_basis=int(nio_kwargs.get("trunk_n_basis", args.nio_trunk_n_basis)),
                    fno_modes=int(nio_kwargs.get("fno_modes", args.nio_fno_modes)),
                    fno_width=int(nio_kwargs.get("fno_width", args.nio_fno_width)),
                    fno_n_layers=int(nio_kwargs.get("fno_n_layers", args.nio_fno_n_layers)),
                    branch_encoder_cls=branch_encoder_cls,
                    branch_encoder_kwargs=branch_encoder_kwargs,
                )
            elif model_key == "inversionnet":
                model = build_inversionnet(
                    dim0=int(cfg_kwargs.get("dim0", 64)),
                    dim1=int(cfg_kwargs.get("dim1", 64)),
                    dim2=int(cfg_kwargs.get("dim2", 64)),
                    dim3=int(cfg_kwargs.get("dim3", 128)),
                    dim4=int(cfg_kwargs.get("dim4", 256)),
                    dim5=int(cfg_kwargs.get("dim5", 512)),
                    regularization=cfg_kwargs.get("regularization", ["l2", 3e-6]),
                )
            else:
                raise ValueError(f"Unsupported model key: {model_key}")

            param_rows, layer_rows, totals = count_model_parameters(model)

            save_csv(
                config_output_dir / f"{model_key}_param_details.csv",
                param_rows,
                ["param_name", "layer_name", "shape", "numel", "trainable"],
            )
            save_csv(
                config_output_dir / f"{model_key}_layer_counts.csv",
                layer_rows,
                ["layer_name", "trainable_params"],
            )

            summary_payload = {
                "model": model_key,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "totals": totals,
                "config_tag": config_tag,
                "config_source": config_source if config_source else "cli",
            }
            with (config_output_dir / f"{model_key}_summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary_payload, f, ensure_ascii=False, indent=2)

            all_totals.append(
                {
                    "config_tag": config_tag,
                    "config_source": config_source if config_source else "cli",
                    "model": model_key,
                    "total_params": totals["total_params"],
                    "total_trainable_params": totals["total_trainable_params"],
                    "total_non_trainable_params": totals["total_non_trainable_params"],
                }
            )

            print(
                f"[{config_tag}::{model_key}] total={totals['total_params']:,}, "
                f"trainable={totals['total_trainable_params']:,}, "
                f"non_trainable={totals['total_non_trainable_params']:,}"
            )

    save_csv(
        output_dir / "all_models_totals.csv",
        all_totals,
        ["config_tag", "config_source", "model", "total_params", "total_trainable_params", "total_non_trainable_params"],
    )

    print(f"Saved parameter reports to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

