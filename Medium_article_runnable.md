# Comparing Nine Sequence Models for Electricity Forecasting

*Sequence Models for Prediction — Part 15*

Hourly household power consumption has daily patterns, irregular spikes, and missing
measurements. We compare nine models on the same task: predict the next 24 hours
from the previous 168 hourly observations.

The input contains only past demand. Calendar variables, weather, and other electrical
measurements are excluded. This keeps the input information consistent across models.
The comparison measures forecast error and model size.

## Data source and preprocessing

The data comes from the [UCI Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
It contains 2,075,259 minute-level measurements from one home in Sceaux, France,
recorded between December 2006 and November 2010. The [original ZIP](https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip)
contains `household_power_consumption.txt`.

We select `Global_active_power` and average the available minute readings within
each clock hour. The result has 34,589 hourly observations. The target is mean
active power in kilowatts, not the sum of the minute-level values.

There are 421 hours with no valid readings. Those hours are forward-filled from
the preceding observed hour. No future observation is used to fill a gap. The
first and last clock hours are partial and are retained. Timestamps keep their
original local-clock labels without a timezone conversion.

![Daily mean power and three weeks of hourly demand](medium-assets/electricity_data.png)

*The upper panel shows the chronological split. The lower panel shows the variation
within a three-week period.*

## Forecasting task and evaluation

At forecast origin `t`, each model uses the preceding week and predicts the next day:

```python
X = demand[t - 168:t]  # observed history
y = demand[t:t + 24]   # 24 future hourly values
```

Every implementation receives a tensor with shape `[batch, 168, 1]` and returns
`[batch, 24]`. The entire future path is predicted directly rather than recursively
feeding predictions back into the model.

The series is split chronologically into 70% training, 15% validation, and 15% test
periods. Every target window stays within one partition. Validation and test inputs
can use earlier observed history from the preceding partition. Standardization
statistics are fitted on the training period only.

This gives 24,021 training origins, 5,165 validation origins, and 5,166 test origins.
Origins are one hour apart, so their 24-hour targets overlap and forecast errors
are correlated.

The primary metric is RMSE in kW, calculated over all test origins and horizons
after reversing standardization. It includes forward-filled target hours. A second
score excludes target hours that had no valid minute readings.

## Model configurations

The linear model maps the flattened input window directly to 24 outputs. The MLP
uses two hidden layers with 256 and 128 units. The GRU and LSTM each use one
64-unit recurrent layer and a final prediction head.

CNN1D uses three convolutional layers followed by global average pooling. The TCN
uses four causal convolutions with kernel size 3 and dilations 1, 2, 4, and 8.
Its final state therefore has a 31-hour receptive field, which is shorter than
the 168-hour input.

The Transformer uses 64-dimensional tokens, learned positions, two encoder blocks,
four attention heads, and feed-forward width 128. The patch version groups the
week into fourteen nonoverlapping 12-hour patches before applying the encoder.

The N-BEATS-style model has three residual dense blocks, each with three 256-unit
hidden layers and separate backcast and forecast heads. It is a generic residual
implementation rather than a full reproduction of the original research model.

All models use AdamW with learning rate 0.001, weight decay 0.0001, batch size 256,
and MSE on standardized targets. Training stops after at most 15 epochs or three
epochs without validation improvement. The checkpoint with the lowest validation
loss is restored before test evaluation. This run uses one seed, 42.

Three additional baselines copy yesterday's profile, last week's profile, or the
last observed value across the forecast horizon.

## Results

| Model | Test RMSE (kW) | Parameters |
|---|---:|---:|
| LSTM | 0.5864 | 18,712 |
| GRU | 0.5876 | 14,424 |
| N-BEATS-style | 0.5882 | 672,576 |
| MLP | 0.5886 | 79,256 |
| Linear | 0.5916 | 4,056 |
| Patch Transformer | 0.5935 | 70,360 |
| Transformer | 0.5957 | 79,512 |
| TCN | 0.6039 | 38,872 |
| CNN1D | 0.6267 | 32,600 |
| Yesterday | 0.7496 | — |
| Last week | 0.8025 | — |
| Last observed value | 0.9179 | — |

*Lower RMSE is better. Parameter counts refer to trainable model parameters.*

LSTM has the lowest test RMSE at **0.5864 kW**. GRU, N-BEATS-style,
and MLP are close, at 0.5876,
0.5882, and 0.5886 kW.
The small differences do not establish a stable ordering from a single run.

All nine learned models beat the three naive baselines. Copying yesterday gives
an RMSE of 0.7496 kW, so LSTM reduces that
error by 21.78%.

![RMSE on the processed series and observed target hours](medium-assets/model_comparison.png)

*Bars include forward-filled targets. Orange markers exclude target hours with
no observed minute readings. Both metrics use the same model predictions.*

LSTM's observed-target-only RMSE is 0.5935 kW. The increase
from the primary score shows that imputed hours affect the reported accuracy.

## Accuracy and model size

The linear model reaches 0.5916 kW with 4,056 parameters. Its
RMSE is 0.89% higher than LSTM's. It remains competitive with models
that use recurrence, attention, or multiple nonlinear layers.

The two recurrent models are also relatively compact: LSTM has 18,712 parameters
and GRU has 14,424. The N-BEATS-style network uses 672,576 parameters for a similar
test error. Parameter count does not measure every deployment cost, but the
difference is relevant when memory or model size matters.

The attention models do not improve on the recurrent models in these configurations.
That result is specific to this dataset, input length, and training budget. It does
not establish how the architectures would compare with more data or additional variables.

## Limits of this comparison

The TCN receives the same 168-hour tensor but its prediction can depend on only
the most recent 31 hours. The CNN's global pooling removes exact positions, and
its validation error is still decreasing at epoch 15. Their results combine
architectural choices with limits imposed by these configurations.

This is one household, one split, one seed, and a fixed training schedule. Model
capacity is not matched, and no architecture-specific hyperparameter search is
performed. Boosted-tree models are not included.
Multiple seeds, rolling evaluation periods, a wider TCN receptive field, and a
boosted-tree baseline would provide a more complete comparison.

The full [Jupyter notebook](electricity_models_demand_only.ipynb) includes the data snapshot,
all model definitions, executed outputs, learning curves, and per-horizon errors.
The accompanying files include the processed CSV, source checksums, predictions,
and saved checkpoints.

## Data reference

Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption*
[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54.
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The data was changed by selecting global active power, averaging by hour,
forward-filling empty hours, and constructing forecasting windows.

---

## Runnable code

The article above is unchanged. The cells below contain the complete comparison,
including data loading and all nine model implementations. Copy the Python blocks
in **this section** into Jupyter and run them in order. The short window example
in the article is illustrative and is not an additional execution step.

Install the dependencies in the notebook environment first, then restart its kernel:

```bash
python -m pip install numpy pandas matplotlib torch jupyterlab ipython
```

Upload the supplied `electricity_hourly.csv` to your public GitHub repository's
`data/` folder. In the first cell, replace `YOUR_USERNAME` and `YOUR_REPO` in
`DATA_URL` with your details. The loader uses this call:

```python
hourly = pd.read_csv(DATA_URL, parse_dates=["timestamp"], index_col="timestamp")
```

The line above is a preview of the loader in cell 2, not a separate setup step.
If the GitHub URL has not been configured, cell 2 uses an existing local CSV or
downloads and processes the official UCI archive automatically. No other project
Python file or notebook is required.

`REQUESTED_DEVICE = "auto"` selects an available supported accelerator or CPU.
For a specific device, use `cpu`, `mps`, `cuda`, `xpu`, `tpu`, or `xla`.
Mac MPS requires a compatible macOS/PyTorch installation. CUDA/ROCm/XPU require
their matching PyTorch builds and drivers. For native Cloud TPU, install
`torch_tpu` on a TPU host before starting the notebook; for an existing XLA TPU
environment, install matching `torch` and `torch_xla[tpu]` versions separately.
Installation instructions: [PyTorch](https://pytorch.org/get-started/locally/) and
[Cloud TPU](https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu).

The saved article scores came from the documented CPU run. The accelerator paths
were not hardware-tested here. This code uses one selected device and includes
a forward/backward/update preflight for every model.
Training time is excluded from the article and main comparison. Set
`REPORT_TIMING = True` to measure fit times on your machine, including macOS/MPS.
The optional report records hardware details and synchronizes the device before
and after each fit. Library versions and hardware can change exact scores. The code writes new
results to `outputs/` and regenerates the article images in `medium-assets/`.

### Cell 1: Imports, settings, and device selection

```python
import base64
import copy
import gzip
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import random
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Set before importing torch.
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from IPython.display import display

DATA_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/data/electricity_hourly.csv"
LOOKBACK = 168
HORIZON = 24
SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 15
PATIENCE = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
REPORT_TIMING = False  # Set True to measure/report fit time on your own hardware.
MODEL_NAMES = ["Linear", "MLP", "GRU", "LSTM", "CNN1D", "TCN",
               "Patch Transformer", "N-BEATS-style", "Transformer"]
REQUESTED_DEVICE = "auto"  # auto / cpu / cuda / mps / xpu / tpu / xla; cuda:1 also works.
torch_xla = xm = None

def select_device(requested="auto"):
    global torch_xla, xm
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")  # NVIDIA CUDA or AMD ROCm PyTorch build.
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            accelerator = torch.accelerator.current_accelerator()
            if accelerator is not None and accelerator.type == "tpu":
                return torch.device("tpu")
        return torch.device("cpu")
    if requested in ("tpu", "xla"):
        if requested == "tpu":
            try:
                import torch_tpu  # Optional native TPU backend; installed only on a TPU host.
            except ImportError as error:
                raise RuntimeError("Install torch_tpu on a Cloud TPU host, restart the kernel, "
                                   "and select REQUESTED_DEVICE='tpu'.") from error
            return torch.device("tpu")
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
        except ImportError as error:
            raise RuntimeError("For legacy TPU/XLA, install matching torch and torch_xla[tpu] "
                               "versions in a separate TPU environment and restart.") from error
        return torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
    kind = requested.split(":")[0]
    available = {"cpu": True, "cuda": torch.cuda.is_available(),
                 "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
                 "xpu": hasattr(torch, "xpu") and torch.xpu.is_available()}
    if kind not in available or not available[kind]:
        raise RuntimeError(f"Device {requested!r} is unavailable in this PyTorch installation. "
                           "Use 'auto'/'cpu' or install the build for your hardware.")
    return torch.device(requested)

DEVICE = select_device(REQUESTED_DEVICE)

def synchronize_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    elif DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "xpu":
        torch.xpu.synchronize(DEVICE)
    elif DEVICE.type == "tpu":
        if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
            torch.accelerator.synchronize()
        else:
            torch.tpu.synchronize()
    elif DEVICE.type == "xla":
        if hasattr(torch_xla, "sync"):
            torch_xla.sync(wait=True)
        else:
            xm.mark_step()
            xm.wait_device_ops()

def optimizer_step(optimizer):
    optimizer.step()
    if DEVICE.type == "xla":
        # XLA is lazy; materialize each optimizer update explicitly.
        if hasattr(torch_xla, "sync"):
            torch_xla.sync()
        else:
            xm.mark_step()

def clear_device_cache():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "xpu" and hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()
torch.set_num_threads(min(4, os.cpu_count() or 1))
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "weights").mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if DEVICE.type == "mps":
        torch.mps.manual_seed(seed)
    elif DEVICE.type == "xpu":
        torch.xpu.manual_seed_all(seed)
    elif DEVICE.type == "tpu":
        torch.tpu.manual_seed_all(seed)
    elif DEVICE.type == "xla":
        xm.set_rng_state(seed, str(DEVICE))

def hardware_summary():
    system = platform.system()
    processor = platform.processor() or platform.machine()
    if system == "Darwin":
        try:
            probe = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=False, timeout=5)
            processor = probe.stdout.strip() or processor
        except (OSError, subprocess.SubprocessError):
            pass
    accelerator = "CPU"
    if DEVICE.type == "cuda":
        accelerator = torch.cuda.get_device_name(DEVICE)
    elif DEVICE.type == "mps":
        accelerator = f"Apple Metal / {processor}"
    elif DEVICE.type == "xpu":
        accelerator = torch.xpu.get_device_name(DEVICE)
    elif DEVICE.type in ("tpu", "xla"):
        accelerator = f"TPU backend ({DEVICE.type})"
    return {"operating_system": "macOS" if system == "Darwin" else system,
            "os_version": platform.mac_ver()[0] if system == "Darwin" else platform.release(),
            "architecture": platform.machine(), "processor": processor,
            "device": str(DEVICE), "accelerator": accelerator,
            "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(),
            "dtype": "float32", "batch_size": BATCH_SIZE,
            "mps_cpu_fallback_enabled": DEVICE.type == "mps" and
                os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"}

def write_local_timing_report(results):
    if not REPORT_TIMING:
        return
    local = results.loc[results["kind"] == "Learned",
        ["model", "training_seconds", "epochs_run", "best_epoch"]].copy()
    local = local.rename(columns={"training_seconds": "fit_wall_seconds"})
    hardware = hardware_summary()
    for key in ("device", "operating_system", "accelerator", "batch_size"):
        local[key] = hardware[key]
    local.to_csv(OUTPUT_DIR / "local_timing.csv", index=False)
    details = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "timing_scope": "training + validation + checkpoint and history writes",
        "excluded": "model initialization, best-checkpoint restore, final test prediction",
        "first_use_compilation": "included when required by the selected backend",
        "synchronization": "selected accelerator synchronized before and after the fit loop",
        "interpretation": "local diagnostics; no comparison across different machines"}
    (OUTPUT_DIR / "local_timing_metadata.json").write_text(json.dumps(details, indent=2))
    print("Local fit times on this machine (training + validation + checkpoint writes):")
    display(local.round({"fit_wall_seconds": 2}))
    print(json.dumps(hardware, indent=2))

seed_everything()
environment = {"python": platform.python_version(), "numpy": np.__version__,
               "pandas": pd.__version__, "torch": torch.__version__,
               "matplotlib": matplotlib.__version__, "device": str(DEVICE),
               "cpu_threads": torch.get_num_threads(), "requested_device": REQUESTED_DEVICE,
               "hip_version": getattr(torch.version, "hip", None)}
print(json.dumps(environment, indent=2))
```

### Cell 2: Load the CSV from GitHub

```python
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption"
RAW_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
RAW_ZIP_SHA256 = "9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff"
HOURLY_CSV_SHA256 = "afbfc86aad2fbc2b0466fab43ea7bf58d089b2d287e229dbb883eafaf8e8c758"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
HOURLY_CSV = DATA_DIR / "electricity_hourly.csv"
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()

def rebuild_hourly_from_uci():
    archive = CACHE_DIR / "household_power_consumption.zip"
    if not archive.exists():
        print("Downloading the official UCI archive (~20 MB):", RAW_URL)
        temporary = archive.with_suffix(".zip.part")
        urllib.request.urlretrieve(RAW_URL, temporary)
        temporary.replace(archive)
    raw_zip = archive.read_bytes()
    if sha256_bytes(raw_zip) != RAW_ZIP_SHA256:
        raise ValueError("UCI archive checksum changed. Inspect the source before proceeding.")
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        with z.open("household_power_consumption.txt") as f:
            minute = pd.read_csv(f, sep=";", na_values=["?"],
                                 usecols=["Date", "Time", "Global_active_power"])
    timestamps = pd.to_datetime(minute["Date"] + " " + minute["Time"],
                                format="%d/%m/%Y %H:%M:%S")
    raw = pd.Series(minute["Global_active_power"].to_numpy(), index=timestamps,
                    name="demand_kw")
    assert raw.index.is_monotonic_increasing and raw.index.is_unique
    hourly = raw.resample("h").agg(["mean", "count", "size"])
    hourly = hourly.rename(columns={"mean": "demand_kw_observed",
                                      "count": "valid_minutes", "size": "recorded_minutes"})
    hourly["demand_kw"] = hourly["demand_kw_observed"].ffill()
    hourly["was_imputed"] = hourly["demand_kw_observed"].isna()
    # Drop only leading hours with no previous observation. This archive has none.
    hourly = hourly.loc[hourly["demand_kw"].notna(),
                          ["demand_kw", "demand_kw_observed", "valid_minutes",
                           "recorded_minutes", "was_imputed"]]
    hourly.index.name = "timestamp"
    csv_content = hourly.to_csv(float_format="%.10g", lineterminator="\n").encode("utf-8")
    if sha256_bytes(csv_content) != HOURLY_CSV_SHA256:
        raise ValueError("The rebuilt hourly CSV differs from the supplied snapshot.")
    HOURLY_CSV.write_bytes(csv_content)
    raw_audit = {"minute_rows": len(raw), "missing_minute_readings": int(raw.isna().sum()),
                 "first_minute": str(raw.index[0]), "last_minute": str(raw.index[-1])}
    print("Raw data audit:", raw_audit)
    return csv_content

# After uploading the supplied CSV to GitHub, replace DATA_URL in the first cell.
# Until then, the same data is loaded locally or rebuilt from the official UCI ZIP.
github_url_configured = bool(DATA_URL) and "YOUR_USERNAME" not in DATA_URL and "YOUR_REPO" not in DATA_URL
if github_url_configured:
    hourly = pd.read_csv(DATA_URL, parse_dates=["timestamp"], index_col="timestamp")
    required_columns = {"demand_kw", "demand_kw_observed", "valid_minutes", "recorded_minutes", "was_imputed"}
    missing_columns = required_columns.difference(hourly.columns)
    if missing_columns:
        raise ValueError(f"Use the supplied electricity_hourly.csv; missing columns: {sorted(missing_columns)}")
    # Normalize line endings while retaining the published CSV's numeric precision.
    csv_content = hourly.to_csv(float_format="%.10g", lineterminator="\n").encode("utf-8")
    if sha256_bytes(csv_content) != HOURLY_CSV_SHA256:
        raise ValueError("The GitHub CSV differs from the dataset used for the article.")
    HOURLY_CSV.write_bytes(csv_content)
    print("Loaded hourly CSV:", DATA_URL)
else:
    csv_content = HOURLY_CSV.read_bytes() if HOURLY_CSV.exists() else rebuild_hourly_from_uci()
    if sha256_bytes(csv_content) != HOURLY_CSV_SHA256:
        raise ValueError("The local hourly CSV differs from the dataset used for the article.")
    hourly = pd.read_csv(io.BytesIO(csv_content), parse_dates=["timestamp"]).set_index("timestamp")
    print("Using the supplied local CSV or the official UCI download.")

assert len(hourly) == 34589
assert hourly.index.is_unique and hourly.index.is_monotonic_increasing
assert (hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
assert hourly["demand_kw"].notna().all()
assert (hourly["was_imputed"] == hourly["demand_kw_observed"].isna()).all()
assert np.allclose(hourly["demand_kw"], hourly["demand_kw_observed"].ffill())

audit = {"dataset": DATASET_PAGE, "raw_download": RAW_URL,
         "doi": "10.24432/C58K54", "license": "CC BY 4.0",
         "raw_zip_sha256": RAW_ZIP_SHA256, "hourly_csv_sha256": HOURLY_CSV_SHA256,
         "raw_rows": 2075259, "missing_minute_readings": 25979,
         "hourly_rows": len(hourly), "first_hour": str(hourly.index[0]),
         "last_hour": str(hourly.index[-1]),
         "forward_filled_hours": int(hourly["was_imputed"].sum()),
         "partial_edge_hours": int((hourly["recorded_minutes"] < 60).sum()),
         "units": "kW", "timestamp_convention": "naive local clock; interval start",
         "preprocessing": "hourly mean of available minute readings; forward fill empty hours"}
(DATA_DIR / "source_metadata.json").write_text(json.dumps(audit, indent=2))
display(pd.Series(audit).rename("value").to_frame())
display(hourly.head())
```

### Cell 3: Chronological split and training-only scaling

```python
n = len(hourly)
train_end = int(0.70 * n)
val_end = int(0.85 * n)
origin_sets = {
    "Train": np.arange(LOOKBACK, train_end - HORIZON + 1),
    "Validation": np.arange(train_end, val_end - HORIZON + 1),
    "Test": np.arange(val_end, n - HORIZON + 1),
}
for name, origins in origin_sets.items():
    assert len(origins) and origins[0] >= LOOKBACK
    assert origins[-1] + HORIZON <= {"Train": train_end, "Validation": val_end, "Test": n}[name]
assert origin_sets["Train"][-1] + HORIZON <= origin_sets["Validation"][0]
assert origin_sets["Validation"][-1] + HORIZON <= origin_sets["Test"][0]

values = hourly["demand_kw"].to_numpy(dtype=np.float64)
observed = hourly["valid_minutes"].to_numpy() > 0
train_mean = values[:train_end].mean()
train_std = values[:train_end].std(ddof=0)
assert train_std > 0
z = ((values - train_mean) / train_std).astype(np.float32)

split_table = pd.DataFrame([
    {"partition": name, "origins": len(origins),
     "first_origin": hourly.index[origins[0]],
     "last_origin": hourly.index[origins[-1]],
     "last_target": hourly.index[origins[-1] + HORIZON - 1]}
    for name, origins in origin_sets.items()
]).set_index("partition")
display(split_table)
print(f"Training-only scaler: mean={train_mean:.6f} kW; std={train_std:.6f} kW")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
daily = hourly["demand_kw"].resample("D").mean()
axes[0].plot(daily.index, daily, lw=0.7, color="#2563eb")
axes[0].axvline(hourly.index[train_end], color="#d97706", ls="--", label="Validation starts")
axes[0].axvline(hourly.index[val_end], color="#dc2626", ls="--", label="Test starts")
axes[0].set(title="Daily mean of the hourly power series", ylabel="Mean power (kW)")
axes[0].legend(loc="upper right")
sample = hourly.loc["2007-01-01":"2007-01-21 23:00", "demand_kw"]
axes[1].plot(sample.index, sample, lw=0.7, color="#0f766e")
axes[1].set(title="Three weeks of hourly household demand", ylabel="Power (kW)")
fig.savefig(OUTPUT_DIR / "electricity_data.png", bbox_inches="tight")
plt.show()
```

### Cell 4: Create the input and target windows

```python
class ForecastDataset(Dataset):
    def __init__(self, standardized_values, origins):
        self.values = torch.as_tensor(standardized_values, dtype=torch.float32)
        self.origins = np.asarray(origins)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        t = int(self.origins[index])
        return self.values[t-LOOKBACK:t, None], self.values[t:t+HORIZON]

datasets = {name: ForecastDataset(z, origins) for name, origins in origin_sets.items()}

def make_loader(name, shuffle=False):
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(datasets[name], batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, generator=generator,
                      pin_memory=DEVICE.type == "cuda")

validation_loader = make_loader("Validation")
test_loader = make_loader("Test")
xb, yb = next(iter(make_loader("Train")))
print("X:", tuple(xb.shape), "y:", tuple(yb.shape))
assert xb.shape[1:] == (LOOKBACK, 1) and yb.shape[1:] == (HORIZON,)

test_origins = origin_sets["Test"]
y_test_kw = np.stack([values[t:t+HORIZON] for t in test_origins])
observed_test_mask = np.stack([observed[t:t+HORIZON] for t in test_origins])
print(f"Observed target cells: {observed_test_mask.mean():.2%}")
```

### Cell 5: Evaluate the three baseline forecasts

```python
def score_predictions(prediction_kw):
    assert prediction_kw.shape == y_test_kw.shape
    assert np.isfinite(prediction_kw).all()
    error = np.asarray(prediction_kw, dtype=np.float64) - y_test_kw
    return {"test_rmse_kw": float(np.sqrt(np.mean(error**2))),
            "observed_rmse_kw": float(np.sqrt(np.mean(error[observed_test_mask]**2))),
            "test_mae_kw": float(np.mean(np.abs(error)))}

baseline_predictions = {
    "Yesterday": np.stack([values[t-24:t] for t in test_origins]),
    "Last week": np.stack([values[t-168:t-144] for t in test_origins]),
    "Last observed value": np.stack([np.repeat(values[t-1], HORIZON) for t in test_origins]),
}
baseline_rows = [{"model": name, "kind": "Naive", **score_predictions(prediction),
                  "parameters": 0, "epochs_run": 0, "best_epoch": 0,
                  "validation_rmse_kw": np.nan, "training_seconds": 0.0}
                 for name, prediction in baseline_predictions.items()]
baselines = pd.DataFrame(baseline_rows).sort_values("test_rmse_kw")
display(baselines[["model", "test_rmse_kw", "observed_rmse_kw", "test_mae_kw"]].round(6))

article_baselines = {"Yesterday": 0.7496, "Last week": 0.8025, "Last observed value": 0.9179}
for row in baseline_rows:
    assert abs(row["test_rmse_kw"] - article_baselines[row["model"]]) < 0.00005
print("All three baseline scores agree with the article to four decimals.")
```

### Cell 6: Define all nine models

```python
class LinearForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(LOOKBACK, HORIZON))

    def forward(self, x):
        return self.net(x)


class MLPForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(LOOKBACK, 256), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(128, HORIZON))

    def forward(self, x):
        return self.net(x)


class CNNForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(64, HORIZON)

    def forward(self, x):
        return self.head(self.features(x.transpose(1, 2)).squeeze(-1))


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TCNForecast(nn.Module):
    def __init__(self, dilations=(1, 2, 4, 8)):
        super().__init__()
        layers, channels = [], 1
        for dilation in dilations:
            layers += [CausalConv1d(channels, 64, 3, dilation), nn.ReLU(), nn.Dropout(0.1)]
            channels = 64
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(64, HORIZON)
        self.receptive_field = 1 + 2 * sum(dilations)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2))[:, :, -1])


class RecurrentForecast(nn.Module):
    def __init__(self, kind):
        super().__init__()
        recurrent_class = {"GRU": nn.GRU, "LSTM": nn.LSTM}[kind]
        self.recurrent = recurrent_class(input_size=1, hidden_size=64, batch_first=True)
        self.head = nn.Linear(64, HORIZON)

    def forward(self, x):
        sequence, _ = self.recurrent(x)
        return self.head(sequence[:, -1])


class TransformerForecast(nn.Module):
    def __init__(self, patch_size=1):
        super().__init__()
        assert LOOKBACK % patch_size == 0
        self.patch_size = patch_size
        token_count = LOOKBACK // patch_size
        self.project = nn.Linear(patch_size, 64)
        self.positions = nn.Parameter(torch.randn(1, token_count, 64) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                           dropout=0.1, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(64),
                                             enable_nested_tensor=False)
        self.head = nn.Linear(64, HORIZON)

    def forward(self, x):
        tokens = x.squeeze(-1).reshape(x.shape[0], LOOKBACK // self.patch_size, self.patch_size)
        encoded = self.encoder(self.project(tokens) + self.positions)
        # Attention sees only the observed history, so no future-target mask is needed.
        return self.head(encoded.mean(dim=1))


class NBeatsBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(LOOKBACK, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU())
        self.backcast = nn.Linear(256, LOOKBACK)
        self.forecast = nn.Linear(256, HORIZON)

    def forward(self, residual):
        hidden = self.hidden(residual)
        return self.backcast(hidden), self.forecast(hidden)


class NBeatsForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([NBeatsBlock() for _ in range(3)])

    def forward(self, x):
        residual = x.squeeze(-1)
        forecast = x.new_zeros((x.shape[0], HORIZON))
        for block in self.blocks:
            backcast, increment = block(residual)
            residual = residual - backcast
            forecast = forecast + increment
        return forecast


MODEL_FACTORIES = {
    "Linear": LinearForecast,
    "MLP": MLPForecast,
    "CNN1D": CNNForecast,
    "TCN": TCNForecast,
    "GRU": lambda: RecurrentForecast("GRU"),
    "LSTM": lambda: RecurrentForecast("LSTM"),
    "Transformer": TransformerForecast,
    "Patch Transformer": lambda: TransformerForecast(patch_size=12),
    "N-BEATS-style": NBeatsForecast,
}

def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

architecture_rows = []
for name in MODEL_NAMES:
    seed_everything()
    model = MODEL_FACTORIES[name]().eval()
    with torch.no_grad():
        output = model(xb[:2])
    assert output.shape == (2, HORIZON) and torch.isfinite(output).all()
    architecture_rows.append({"model": name, "parameters": parameter_count(model)})
display(pd.DataFrame(architecture_rows).set_index("model"))
print("All nine models satisfy the input/output contract.")
del model
```

### Cell 7: Check the selected device before training

```python
# One actual forward/backward/update per architecture on the selected backend.
# This uses fresh throwaway models; it does not modify any benchmark checkpoints.
def device_preflight():
    checks = []
    for name in MODEL_NAMES:
        seed_everything()
        probe = MODEL_FACTORIES[name]().to(DEVICE).train()
        optimizer = torch.optim.AdamW(probe.parameters(), lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            foreach=False if DEVICE.type in ("mps", "tpu", "xla") else None)
        x, y = xb[:2].to(DEVICE), yb[:2].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        prediction = probe(x)
        loss = F.mse_loss(prediction, y)
        loss.backward()
        nn.utils.clip_grad_norm_(probe.parameters(), 1.0,
            foreach=False if DEVICE.type in ("mps", "tpu", "xla") else None)
        optimizer_step(optimizer)
        synchronize_device()
        if prediction.shape != y.shape or not bool(torch.isfinite(loss.detach()).cpu().item()):
            raise RuntimeError(f"Invalid preflight output for {name} on {DEVICE}")
        checks.append({"model": name, "device": str(DEVICE), "forward_backward_update": "passed"})
        del probe, optimizer
    return pd.DataFrame(checks)

try:
    preflight_results = device_preflight()
except (NotImplementedError, RuntimeError) as error:
    unsupported = isinstance(error, NotImplementedError) or any(
        word in str(error).lower() for word in ("not implemented", "not supported", "unsupported"))
    if REQUESTED_DEVICE == "auto" and DEVICE.type != "cpu" and unsupported:
        print(f"{DEVICE} failed the model preflight: {error}")
        print("Using CPU for all nine models in this run.")
        DEVICE = torch.device("cpu")
        environment["device"] = str(DEVICE)
        preflight_results = device_preflight()
    else:
        raise RuntimeError(f"Model preflight failed on {DEVICE}. Choose 'cpu' for the portable "
                           "reference run or resolve this backend/operator error.") from error
display(preflight_results)
clear_device_cache()
seed_everything()
```

### Cell 8: Training and evaluation functions

```python
@torch.inference_mode()
def validation_mse(model):
    model.eval()
    squared_error, elements = 0.0, 0
    for x, y in validation_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        prediction = model(x)
        squared_error += F.mse_loss(prediction, y, reduction="sum").item()
        elements += y.numel()
    return squared_error / elements


@torch.inference_mode()
def predict_test_kw(model):
    model.eval()
    predictions = [model(x.to(DEVICE)).cpu().numpy() for x, _ in test_loader]
    return np.concatenate(predictions).astype(np.float64) * train_std + train_mean


def train_model(name):
    seed_everything()
    model = MODEL_FACTORIES[name]().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY,
                                  foreach=False if DEVICE.type in ("mps", "tpu", "xla") else None)
    train_loader = make_loader("Train", shuffle=True)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state, history = None, []
    checkpoint = OUTPUT_DIR / "weights" / (name.lower().replace(" ", "_") + ".pt")
    started = None
    if REPORT_TIMING:
        synchronize_device()
        started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum, elements = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = F.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {name}, epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                     foreach=False if DEVICE.type in ("mps", "tpu", "xla") else None)
            optimizer_step(optimizer)
            loss_sum += loss.item() * y.numel()
            elements += y.numel()

        val_loss = validation_mse(model)
        row = {"epoch": epoch, "train_rmse_kw": math.sqrt(loss_sum / elements) * train_std,
               "validation_rmse_kw": math.sqrt(val_loss) * train_std}
        history.append(row)
        print(f"{name:18s} {epoch:02d}/{MAX_EPOCHS} | "
              f"train RMSE {row['train_rmse_kw']:.4f} | "
              f"validation RMSE {row['validation_rmse_kw']:.4f}", flush=True)

        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            torch.save(best_state, checkpoint)
        else:
            stale += 1
        pd.DataFrame(history).to_csv(checkpoint.with_suffix(".history.csv"), index=False)
        if stale >= PATIENCE:
            print(f"Early stopping; restoring epoch {best_epoch}.", flush=True)
            break

    training_seconds = None
    if REPORT_TIMING:
        synchronize_device()
        training_seconds = time.perf_counter() - started
    assert best_state is not None
    model.load_state_dict(best_state)
    prediction_kw = predict_test_kw(model)
    result = {"model": name, "kind": "Learned", **score_predictions(prediction_kw),
              "parameters": parameter_count(model), "epochs_run": len(history),
              "best_epoch": best_epoch, "validation_rmse_kw": math.sqrt(best_loss) * train_std,
              "training_seconds": training_seconds, "device": str(DEVICE)}
    del model, optimizer
    clear_device_cache()
    return result, prediction_kw, history
```

### Cell 9: Train all nine models and save results

```python
learned_rows, predictions_by_model, histories = [], dict(baseline_predictions), {}
run_started = datetime.now(timezone.utc).isoformat()
for name in MODEL_NAMES:
    print(f"\n{'=' * 20} {name} {'=' * 20}", flush=True)
    result, predictions_by_model[name], histories[name] = train_model(name)
    learned_rows.append(result)
    pd.DataFrame(learned_rows).to_csv(OUTPUT_DIR / "learned_results_partial.csv", index=False)
    print(f"Completed {name}: test RMSE = {result['test_rmse_kw']:.6f} kW", flush=True)

results = pd.DataFrame(learned_rows + baseline_rows).sort_values("test_rmse_kw").reset_index(drop=True)
results.index = np.arange(1, len(results) + 1)
results.index.name = "rank"
assert len(learned_rows) == 9 and len(results) == 12
comparison_results = results.drop(columns=["training_seconds", "device"], errors="ignore")
comparison_results.to_csv(OUTPUT_DIR / "model_comparison.csv")
display(comparison_results[["model", "test_rmse_kw", "observed_rmse_kw", "parameters", "epochs_run"]]
        .round({"test_rmse_kw": 6, "observed_rmse_kw": 6}))
write_local_timing_report(results)

run_metadata = {"started_utc": run_started, "completed_utc": datetime.now(timezone.utc).isoformat(),
                "seed": SEED, "lookback": LOOKBACK, "horizon": HORIZON,
                "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": PATIENCE, "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW",
                "gradient_clip_norm": 1.0, "models": MODEL_NAMES,
                "environment": environment, "hardware": hardware_summary(),
                "report_timing": REPORT_TIMING, "data": audit,
                "train_mean_kw": float(train_mean), "train_std_kw": float(train_std),
                "train_end": train_end, "validation_end": val_end,
                "origin_counts": {name: len(origins) for name, origins in origin_sets.items()}}
(OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))
np.savez_compressed(OUTPUT_DIR / "test_predictions.npz", actual_kw=y_test_kw,
                     observed_mask=observed_test_mask, origins=test_origins,
                     **{name.replace(" ", "_"): value for name, value in predictions_by_model.items()})
```

### Cell 10: Plot accuracy, model size, and learning curves

```python
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
chart = results.iloc[::-1]
colors = ["#94a3b8" if kind == "Naive" else "#2563eb" for kind in chart["kind"]]
ax.barh(chart["model"], chart["test_rmse_kw"], color=colors, label="RMSE on processed series")
ax.scatter(chart["observed_rmse_kw"], chart["model"], color="#f59e0b", s=30,
            zorder=3, label="RMSE on observed target hours")
ax.set(xlabel="Test RMSE (kW; lower is better)", title="Nine learned models and three naive baselines")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
ax.grid(axis="x", alpha=0.15)
fig.savefig(OUTPUT_DIR / "model_comparison.png", bbox_inches="tight")
plt.show()

learned = results[results["kind"] == "Learned"]
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.scatter(learned["parameters"], learned["test_rmse_kw"], color="#0f766e", s=45)
for _, row in learned.iterrows():
    ax.annotate(row["model"], (row["parameters"], row["test_rmse_kw"]),
                xytext=(4, 5), textcoords="offset points", fontsize=8)
ax.set(xscale="log", xlabel="Trainable parameters (log scale)", ylabel="Test RMSE (kW)",
        title="Accuracy versus model size")
ax.grid(alpha=0.15)
fig.savefig(OUTPUT_DIR / "accuracy_vs_size.png", bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(3, 3, figsize=(12, 8), constrained_layout=True)
for ax, name in zip(axes.flat, MODEL_NAMES):
    history = pd.DataFrame(histories[name])
    ax.plot(history["epoch"], history["train_rmse_kw"], label="Train", color="#2563eb")
    ax.plot(history["epoch"], history["validation_rmse_kw"], label="Validation", color="#d97706")
    ax.set(title=name, xlabel="Epoch", ylabel="RMSE (kW)")
axes.flat[0].legend()
fig.savefig(OUTPUT_DIR / "learning_curves.png", bbox_inches="tight")
plt.show()
```

### Cell 11: Inspect forecasts and errors by horizon

```python
best_name = learned.iloc[0]["model"]
selected_names = [best_name, "Linear", "Yesterday"]
selected_names = list(dict.fromkeys(selected_names))
fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
for ax, example in zip(axes, [0, len(test_origins) // 2, len(test_origins) - 1]):
    t = int(test_origins[example])
    hours = hourly.index[t:t+HORIZON]
    ax.plot(hours, y_test_kw[example], color="#111827", lw=2, label="Actual (processed series)")
    for name in selected_names:
        ax.plot(hours, predictions_by_model[name][example], lw=1.3, label=name)
    ax.set(ylabel="Power (kW)", title=f"Forecast origin: {hours[0]}")
    ax.legend(ncol=4, fontsize=8)
fig.savefig(OUTPUT_DIR / "forecast_examples.png", bbox_inches="tight")
plt.show()

horizon_rows = []
for name, prediction in predictions_by_model.items():
    rmse = np.sqrt(np.mean((prediction - y_test_kw)**2, axis=0))
    horizon_rows.extend({"model": name, "horizon_hour": int(h+1), "rmse_kw": float(value)}
                        for h, value in enumerate(rmse))
horizon_results = pd.DataFrame(horizon_rows)
horizon_results.to_csv(OUTPUT_DIR / "rmse_by_horizon.csv", index=False)
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
for name in [best_name, "Linear", "Yesterday", "Last week"]:
    group = horizon_results[horizon_results["model"] == name]
    ax.plot(group["horizon_hour"], group["rmse_kw"], label=name)
ax.set(xlabel="Hours ahead", ylabel="RMSE (kW)", title="Forecast error by prediction horizon")
ax.legend()
fig.savefig(OUTPUT_DIR / "rmse_by_horizon.png", bbox_inches="tight")
plt.show()

best = learned.iloc[0]
linear = learned[learned["model"] == "Linear"].iloc[0]
yesterday = baselines[baselines["model"] == "Yesterday"].iloc[0]
improvement = 100 * (1 - best["test_rmse_kw"] / yesterday["test_rmse_kw"])
linear_gap = 100 * (linear["test_rmse_kw"] / best["test_rmse_kw"] - 1)
print(f"Best measured learned model in this run: {best_name}.")
print(f"Test RMSE: {best['test_rmse_kw']:.6f} kW; "
      f"observed-target-only RMSE: {best['observed_rmse_kw']:.6f} kW.")
print(f"RMSE reduction versus yesterday: {improvement:.2f}%.")
print(f"Linear RMSE gap versus the best model: {linear_gap:.2f}%.")
```

### Cell 12: Save figures under the article image paths

```python
import shutil

ASSET_DIR = Path("medium-assets")
ASSET_DIR.mkdir(exist_ok=True)
for filename in ("electricity_data.png", "model_comparison.png", "forecast_examples.png"):
    shutil.copy2(OUTPUT_DIR / filename, ASSET_DIR / filename)

rows = []
for _, row in results.iterrows():
    rows.append([row["model"], f"{row['test_rmse_kw']:.4f}",
                 f"{int(row['parameters']):,}" if row["kind"] == "Learned" else "—"])
fig, ax = plt.subplots(figsize=(9, 5.3), dpi=180)
ax.axis("off")
table = ax.table(cellText=rows,
    colLabels=["Model", "Test RMSE (kW)", "Parameters"],
    cellLoc="left", colLoc="left", loc="center", colWidths=[0.45, 0.28, 0.27])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.65)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#e2e8f0")
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor("#e2e8f0")
        cell.set_text_props(weight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f8fafc")
ax.set_title("Electricity forecasting: measured test results", fontsize=13, pad=12, loc="left")
fig.tight_layout()
fig.savefig(ASSET_DIR / "model_results_table.png", bbox_inches="tight")
plt.show()
print("Article figures saved to:", ASSET_DIR.resolve())
```

