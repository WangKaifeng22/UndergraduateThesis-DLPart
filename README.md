# UndergraduateThesis-DLPart

Deep learning experiments for ultrasound computed tomography (USCT) speed-of-sound reconstruction.
This repository contains training, evaluation, benchmarking, and HDF5 preprocessing pipelines for multiple model families:

- Fourier-DeepONet
- Fourier-DeepONet-F
- BranchTrunkFlower
- InversionNet
- Neural Inverse Operator (NIO)

## Project Layout

The codebase has been reorganized into module-style directories under `src/`:

- `src/models/`: model definitions
  - `InversionNet.py`
  - `model_BranchTrunkFlower.py`
  - `model_FourierDeepONet.py`
  - `model_FourierDeepONetF.py`
  - `model_NIO.py`
- `src/train/`: training entry points and training helpers
  - `train.py`
  - `train_BranchTrunkFlower.py`
  - `train_inversionnet.py`
  - `train_NIO.py`
  - `training_callbacks.py`
- `src/test/`: unified evaluation and robustness tests
  - `test.py`
  - `multi_model_test.py`
  - `robustness_test.py`
- `src/scripts/`: utility scripts for experiments and analysis
  - `benchmark_inference.py`
  - `check_gradient.py`
  - `count_trainable_params.py`
  - `loss_analyzer.py`
  - `plot_loss.py`
  - `plot_timedata.py`
- `src/utils/`: shared data loading, preprocessing, and helper utilities
  - `data.py`
  - `h5_preprocess.py`
  - `h5_dataset.py`
  - `H5NIODataset.py`
  - `multi_data.py`
  - `utils.py`
  - and other shared modules
- `src/optimizer/`: custom optimizers
  - `muon.py`
  - `soap.py`


## Environment

- Python `>=3.10`
- DeepXDE backend: PyTorch (`DDE_BACKEND=pytorch`)

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Note: if you use CUDA, install a GPU-compatible PyTorch build first, then install the remaining dependencies from `requirements.txt`.

## Data Preparation

This project expects SoS and K-Wave simulation outputs organized by parameter folders. If you need to regenerate the raw data, see:

<https://github.com/WangKaifeng22/DataGenerationForOnePatch.git>

The preprocessing and loading utilities are now located under `src/utils/`.

Typical workflow:

1. Configure dataset paths in your training or preprocessing entry script.
2. Generate the HDF5 cache.
3. Reuse the cached HDF5 files for training and evaluation.

Example preprocessing entry:

```powershell
python src\utils\h5_preprocess.py
```

## Training

### Fourier-DeepONet / Fourier-DeepONet-F

Use `src/train/train.py` as the main training entry.

```powershell
python src\train\train.py
```

### BranchTrunkFlower

Use `src/train/train_BranchTrunkFlower.py`.

```powershell
python src\train\train_BranchTrunkFlower.py
```

### InversionNet

Use `src/train/train_inversionnet.py`.

```powershell
python src\train\train_inversionnet.py
```

### NIO

Before running, update `grid_npy_path` in `src/train/train_NIO.py` or your own launcher.

```powershell
python src\train\train_NIO.py
```

## Evaluation

Use `src/test/test.py` for unified evaluation.

The script can auto-read `model_config.json` next to the checkpoint to infer model settings.
Supported model types:

- `FourierDeepONet`
- `BranchTrunkFlower`
- `InversionNet`
- `NIO`

You can edit the `__main__` block in `src/test/test.py` or import and call `main(...)` from your own script.

For robustness experiments, use `src/test/robustness_test.py`.

- `FourierDeepONet` / `BranchTrunkFlower`: branch noise, trunk noise, and combined noise are supported.
- `InversionNet` / `NIO`: branch noise is supported.
- The script saves per-trial metrics, aggregated summaries, and metric-vs-sigma plots under the chosen result directory.

For multi-model comparison, use `src/test/multi_model_test.py`.

## Inference Benchmark

Use `src/scripts/benchmark_inference.py` for latency/throughput benchmarking without plotting or metric post-processing overhead.

Example:

```powershell
python src\scripts\benchmark_inference.py `
  --model-path C:\path\to\model.pt `
  --model-type InversionNet `
  --batch-size 32 `
  --warmup-iters 20 `
  --measure-iters 100 `
  --timing-scope forward
```

Timing scopes:

- `forward`: measures only the model forward pass on prepared device tensors.
- `end2end`: measures tensor creation, host-to-device transfer, and forward pass together.

Outputs:

- Console summary
- Benchmark report next to the checkpoint, or under `--out-dir` if specified

## Analysis and Utilities

Additional helper scripts are available under `src/scripts/`:

- `count_trainable_params.py`: count trainable / non-trainable parameters and export CSV summaries
- `check_gradient.py`: gradient checking
- `loss_analyzer.py`: loss inspection and aggregation
- `plot_loss.py`: training loss visualization
- `plot_timedata.py`: time-series / timing plots

## Interpretability Experiments

The interpretability experiments are organized under `src/interpretability/phase2/`.

- `run_all.py` can launch the full experiment suite
- `experiment_2_1.py` to `experiment_2_4.py` contain individual experiment steps
- `README.md` in that folder explains the phase-2 workflow

## Reproducibility Tips

- Keep `model_config.json` together with each exported checkpoint.
- Prefer HDF5 lazy loading for large datasets (`src/utils/h5_dataset.py`, `src/utils/H5NIODataset.py`).
- Use fixed random seeds (`set_seed(...)` in `src/utils/utils.py`) for stable comparisons.
