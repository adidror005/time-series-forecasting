## Running on Mac, CPU, GPU, or TPU

Leave `REQUESTED_DEVICE = "auto"` for normal use, or choose a backend explicitly.
All model tensors use float32; the float64 metric calculations stay in NumPy on CPU.

| Environment | Device option | Setup |
|---|---|---|
| Intel Mac / Linux / Windows CPU | `cpu` or `auto` | Standard compatible PyTorch build |
| Apple silicon Mac | `mps` or `auto` | macOS/PyTorch with MPS support; unsupported MPS operators may fall back to CPU |
| NVIDIA GPU | `cuda` or `auto` | PyTorch CUDA build and working driver |
| AMD GPU | `cuda` or `auto` | Supported ROCm build; PyTorch uses the CUDA API name for ROCm |
| Intel GPU | `xpu` or `auto` | PyTorch XPU build and working runtime |
| Cloud TPU, native backend | `tpu` (or detected by `auto`) | Install `torch_tpu` on a TPU host and restart the kernel |
| Existing TPU/XLA environment | `xla` | Matching `torch` / `torch_xla[tpu]` versions; explicit lazy optimizer steps are included |

This notebook uses one selected device. Distributed multi-GPU or TPU-pod training
is not configured. GPU and TPU availability depends on the installed backend and
hardware. The preflight cell checks a forward pass, backward pass, and optimizer
step for every model before the full run. In `auto` mode, an unsupported accelerator
operator triggers a clearly reported CPU fallback for the entire comparison.

The displayed benchmark was executed on CPU. The portability update was checked
with a separate CPU smoke test; no Mac GPU, CUDA, ROCm, XPU, or TPU hardware was
available for verification here. If memory is limited, lower `BATCH_SIZE`, for
example to 64. Restart the kernel after changing packages or MPS fallback settings.

Setup references: [PyTorch installer](https://pytorch.org/get-started/locally/),
[MPS](https://docs.pytorch.org/docs/stable/notes/mps.html),
[ROCm](https://docs.pytorch.org/docs/stable/notes/hip.html),
[Intel XPU](https://docs.pytorch.org/docs/stable/xpu.html),
[native TorchTPU](https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu).
TorchTPU is the current TPU path; XLA is retained for existing environments.

## Optional timing
Training time is omitted from the article and main comparison. Set REPORT_TIMING = True to generate local_timing.csv and local_timing_metadata.json with OS, chip/processor, actual device, PyTorch version, threads, batch size, and epoch counts. MPS/CUDA/XPU/TPU execution is synchronized at timer boundaries. Only CPU timing was tested here; no macOS timings have been supplied.

## CatBoost

The expanded notebook also fits CatBoost. Its configured backend is CPU on every platform, including Mac and TPU hosts. The PyTorch device option controls only the nine neural models. Install `catboost` using the supplied requirements.
