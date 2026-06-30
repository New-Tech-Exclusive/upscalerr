# Upscalerr

Pure-Python real-time game upscaler and frame generation for Windows + NVIDIA GPUs.

## Features

- **Zero-copy capture** via DXCam `__cuda_array_interface__` → PyTorch CUDA views
- **BGRA → RGB fp16** conversion on GPU (no hue/channel swap artifacts)
- **TensorRT** ESPCN spatial upscaling + FlowNet optical flow
- **CUDA Graph** replay for low-latency inference during gameplay
- **OpenGL overlay** presentation with CUDA–GL interop (VRAM-only path)
- **Training scripts** for ESPCN and FlowNet-S with ONNX export

## Requirements

- Windows 10/11
- NVIDIA GPU (tested target: RTX 5060 Ti 8GB)
- CUDA 12.x, cuDNN, TensorRT 10.x
- Python 3.10+

## Install

```bash
cd upscalerr
pip install -r requirements.txt
pip install -e .
```

## Build TensorRT Engines

1. Train or place ONNX models in `models/onnx/`:
   - `espcn_2x.onnx`
   - `flownet.onnx`

2. Build FP16 engines:

```bash
python tools/build_engines.py --onnx-dir models/onnx --engine-dir models/engines
```

## Run

```bash
python -m upscalerr --config configs/default.yaml
# or, after installation:
upscalerr-gui --config configs/default.yaml
```

The GUI opens a Lossless Scaling-style control panel with:

1. A capture target picker for the game window
2. Runtime toggles for frame generation, FPS, and overlay behavior
3. A live status area showing frame, inference, present, and VRAM metrics

Select the game window and click **Start** to launch the overlay.

## Train ESPCN

Run from the project root (`upscalerr/upscalerr/`):

```bash
python training/train_espcn.py --config configs/espcn_train.yaml
# or
python -m training.train_espcn --config configs/espcn_train.yaml
# or after pip install -e .
train-espcn --config configs/espcn_train.yaml
```

Place DIV2K HR images under `data/div2k/train`.

## Train FlowNet

```bash
python training/train_flownet.py --config configs/flownet_train.yaml
```

Provide consecutive frame pairs under `data/frames/train` and `data/frames/val` (use `tools/export_game_frames.py` to capture).

## Architecture

```
DXCam grab (BGRA, GPU) → torch.as_tensor (zero-copy)
  → BGRA→RGB fp16 NCHW → TensorRT ESPCN
  → [optional] TensorRT FlowNet → pyramidal bilinear warp (t=0.5)
  → RGBA uint8 → CUDA–GL PBO → OpenGL fullscreen overlay
```

## Tests

```bash
pytest tests/
```
