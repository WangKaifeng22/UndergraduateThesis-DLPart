# UndergraduateThesis-DLPart

Deep learning experiments for ultrasound computed tomography (USCT) inverse reconstruction.
This repository includes training, evaluation, and HDF5 preprocessing pipelines for three model families:

- Fourier-DeepONet
- Fourier-DeepONet-F
- InversionNet
- Neural Inverse Operator (NIO)

## Project Layout

- `src/my_train.py`: Fourier-DeepONet-F training entry
- `src/train_inversionnet.py`: InversionNet training entry
- `src/train_NIO.py`: NIO training entry
- `src/my_test.py`: unified evaluation script (supports all three model types)
- `src/h5_preprocess.py`: raw dataset -> HDF5 preprocessing
- `src/h5_dataset.py`, `src/H5NIODataset.py`: lazy HDF5 dataset loaders
- `src/Plot_Loss.py`, `src/plot_timedata.py`: analysis and plotting utilities

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

> Note: for CUDA, install a GPU-compatible PyTorch build first if needed, then install the rest from `requirements.txt`.

## Data Preparation

This project expects SoS and K-Wave simulation outputs organized by parameter folders. You can generate the data using the code at <https://github.com/WangKaifeng22/DataGenerationForOnePatch.git>.

Current scripts read dataset paths from Python variables (mainly in `src/my_train.py`):

- `sos_root`
- `kwave_root`
- `cache_h5_path`
- `meta_h5_path`
- `x_params`, `y_params`, `samples_per_config`

After updating those paths, generate HDF5 cache:

```powershell
python src\h5_preprocess.py
```

## Training

### 1) Fourier-DeepONet

```powershell
python src\my_train.py
```

### 2) InversionNet

```powershell
python src\train_inversionnet.py
```

### 3) NIO

Before running, update `grid_npy_path` in `src/train_NIO.py` (or your own launcher).

```powershell
python src\train_NIO.py
```

## Evaluation

Use `src/my_test.py` and point to a saved checkpoint (`.pt`).

The script can auto-read `model_config.json` next to the checkpoint to infer model settings.
Supported model types:

- `FourierDeepONet`
- `InversionNet`
- `NIO`

You can either modify the `__main__` block in `src/my_test.py` or call `main(...)` from your own script.

For Gaussian-noise robustness experiments, use `src/robustness_test.py`.

- `FourierDeepONet` / `BranchTrunkFlower`: supports branch noise, trunk noise, and branch+trunk noise.
- `InversionNet` / `NIO`: supports branch noise only.
- The script saves per-trial metrics, aggregated summaries, and metric-vs-sigma plots under the chosen result directory.

## Inference Benchmark

Use `src/benchmark_inference.py` for latency/throughput benchmarking without mixing in plotting and metric post-processing overhead.

Example:

```bash
cd src
python benchmark_inference.py \
	--model-path /path/to/model.pt \
	--model-type InversionNet \
	--batch-size 32 \
	--warmup-iters 20 \
	--measure-iters 100 \
	--timing-scope forward
```

Timing scopes:

- `forward`: measures only model forward pass on prepared device tensors.
- `end2end`: measures per-batch tensor creation and host->device transfer plus forward pass.

Outputs:

- Console summary
- `benchmark_result/benchmark_report.txt` next to the checkpoint (or `--out-dir` if specified)

## Modules not yet adapted:

The following modules have not been validated for operation and may not work correctly.

- `src/multi_data.py`
- `src/soap.py`
- `src/muon.py`

## Reproducibility Tips

- Keep `model_config.json` together with each exported checkpoint.
- Prefer HDF5 lazy loading for large datasets (`src/h5_dataset.py`, `src/H5NIODataset.py`).
- Use fixed random seeds (`set_seed(...)` in `src/utils.py`) for stable comparisons.

