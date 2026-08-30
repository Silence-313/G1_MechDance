# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation and Setup

```bash
conda create -n gvhmr python=3.10 -y
conda activate gvhmr
pip install -r requirements.txt
pip install -e .
```

Pretrained checkpoints go in `inputs/checkpoints/gvhmr/`. Download the released checkpoint `gvhmr_siga24_release.ckpt` from the project page.

## Common Commands

**Demo (single video):**
```bash
python tools/demo/demo.py --video=<path.mp4> -s        # -s = static camera (skip visual odometry)
```

**Demo (batch folder):**
```bash
python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
```

**Test (reproduce benchmarks):**
```bash
# All three benchmarks
python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt

# Individual
python tools/train.py global/task=gvhmr/test_3dpw exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
```

**Train:**
```bash
python tools/train.py exp=gvhmr/mixed/mixed
```

The `global/task` Hydra override switches between `gvhmr/train` and `gvhmr/test_*` task configs.

## Architecture

### Config System (Hydra)

GVHMR uses **Hydra + hydra-zen** for all configuration. Configs are registered to a `ConfigStore` singleton via `MainStore.store()` calls in `hmr4d/configs/store_gvhmr.py`. Key configs:

- `hmr4d/configs/train.yaml` — master training config
- `hmr4d/configs/demo.yaml` — master demo/inference config
- `hmr4d/configs/exp/gvhmr/mixed/mixed.yaml` — the released experiment (pipeline, datasets, callbacks, network)
- `hmr4d/configs/siga24_release.yaml` — config matching the released checkpoint weights

Hydra overrides are passed as `key=value` pairs on the CLI. `hmr4d/configs/global/task/` defines train/test task variants.

### Core Pipeline

The model has **three stages**:

1. **Preprocessing** (`hmr4d/utils/preproc/`): YOLOv8 bounding-box tracking → ViTPose-H 2D keypoints → HMR2 ViT backbone (image features) → SimpleVO visual odometry (camera rotation)
2. **Denoising Transformer** (`hmr4d/network/gvhmr/relative_transformer.py`): 12-layer Transformer encoder with RoPE attention. Takes 2D poses + camera features + image features, outputs 151-dim latent + camera translation + static confidence.
3. **Decoding** (`hmr4d/model/gvhmr/utils/endecoder.py`): The 151-dim latent decodes to SMPL-X parameters (body pose, betas, global orient in both camera and gravity-view frames, local translation velocity).

### Key Classes

- **`GvhmrPL`** (`hmr4d/model/gvhmr/gvhmr_pl.py`): PyTorch Lightning module for training. Manages the full train/val/test loop, loss computation, and metric logging.
- **`DemoPL`** (`hmr4d/model/gvhmr/gvhmr_pl_demo.py`): Lightweight Lightning module for inference-only use (no training logic).
- **`Pipeline`** (`hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py`): Core `nn.Module` — forward pass through the denoising transformer, loss computation (MSE + root-relative joint error + camera error + 2D reprojection + vertex error + world translation error + static-conf BCE).
- **`NetworkEncoderRoPE`** (`hmr4d/network/gvhmr/relative_transformer.py`): The transformer backbone. Inputs: 2D poses (17×3), CLIFF camera (3), angular velocity (6), image features (1024). Latent dim: 512, 8 heads, 12 layers.
- **`EnDecoder`** (`hmr4d/model/gvhmr/utils/endecoder.py`): Normalizes/denormalizes SMPL-X parameters using precomputed mean/std statistics.
- **`HMR2`** (`hmr4d/network/hmr2/hmr2.py`): ViT-H backbone (32 layers, 1280 embed dim) pretrained on HMR2.0a. Used only for the feature token, not the SMPL head.
- **`SimpleVO`** (`hmr4d/utils/preproc/relpose/simple_vo.py`): Custom visual odometry for camera rotation estimation.

### Data Flow

```
Video frames
  → Tracker (YOLOv8 bbox) → VitPose (2D keypoints) → HMR2 ViT (image features)
  → SimpleVO (camera rotation)
  → Concatenated condition vector
  → NetworkEncoderRoPE (denoising transformer)
  → EnDecoder (decode to SMPL-X)
  → Postprocessing (static joint fix, IK refinement)
  → SMPL-X parameters in world coordinates
```

### Postprocessing

After the transformer outputs SMPL-X parameters, postprocessing (`hmr4d/model/gvhmr/utils/postprocess.py`) applies:
1. Static joint detection and fixing (joints that shouldn't move are clamped)
2. IK refinement to ensure foot-ground contact consistency
3. This postprocessing is used during evaluation but NOT during training

### Dataset Structure

Datasets live in `hmr4d/dataset/`:
- `pure_motion/amass.py` — AMASS (motion-only, no images, used for training)
- `imgfeat_motion/` — image-feature + motion datasets: BEDLAM (synthetic, train), Human3.6M (train), 3DPW (train+test), EMDB (test), RICH (test)

Training uses `ConcatDataset` of multiple sources. Validation uses `CombinedLoader` for sequential evaluation.

### Training Details

- Optimizer: AdamW, scheduler: half-cycle cosine to epoch 200 then constant to 350
- The released `gvhmr_siga24_release.ckpt` was trained on 2×4090 for 420 epochs
- During training, postprocessing is NOT applied (metrics will differ from test-time results)

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch==2.3.0+cu121` | Deep learning framework |
| `lightning==2.3.0` | PyTorch Lightning training loop |
| `hydra-core + hydra-zen` | Configuration management |
| `pytorch3d==0.7.6` | 3D mesh rendering |
| `smplx` | SMPL-X body model |
| `ultralytics` | YOLOv8 bounding box tracking |
| `timm` | ViT backbone utilities |

## Constraints

- `--f_mm` flag specifies full-frame camera focal length in mm for demo (important for metric-scale world coordinates)
- `-s` flag on demo skips visual odometry for static-camera videos
- Requires SMPL-X body model files (downloaded separately)
- No automated tests — evaluation is done via benchmark dataset metrics (MPJPE, PA-MPJPE on 3DPW/RICH/EMDB)
