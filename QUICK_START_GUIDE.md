# Quick Start Guide - Testing RF-DETR-Seg Two-Stage Training

This guide shows how to run quick 1-epoch tests of both Stage 1 (detector) and Stage 2 (segmentation) training using the smallest nano model.

## 🚀 TL;DR - Quick Commands

**Validate Setup:**
```powershell
python validate_setup.py
```

**Stage 1 Test (1 epoch, ~5-10 min):**
```powershell
python train_detector_stage1.py --size nano --epochs 1 --batch 1 --accum 4 --workers 0 --multi_scale 0
```

**Stage 2 Test (1 epoch, ~5-10 min):**
```powershell
python train_seg_stage2.py --size nano --stage1_run runs/detector_nano_384_1 --epochs 1 --batch 1 --accum 2 --workers 0 --multi_scale 0 --run_test 0
```

---

## Prerequisites

### 1. Dataset Structure
Your dataset should be in COCO format with this structure:
```
dataset/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg (images)
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg (images)
└── test/ (optional)
    ├── _annotations.coco.json
    └── *.jpg (images)
```

### 2. Annotations Requirements
- **Stage 1 (Detector)**: Requires bounding boxes only
- **Stage 2 (Segmentation)**: Requires both bounding boxes AND segmentation masks (polygons/RLE)

⚠️ **IMPORTANT**: Ensure your COCO JSON has `"segmentation"` field for Stage 2 training!

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Quick Test: 1 Epoch on Nano Model

### Stage 1: Train Detector Only (Detection Head)

This trains only the object detection head with bounding boxes.

```powershell
python train_detector_stage1.py `
    --size nano `
    --epochs 1 `
    --batch 1 `
    --accum 4 `
    --workers 0 `
    --multi_scale 0
```

#### Parameters Explained:
- `--size nano` - Smallest model (~3M params, 384px resolution)
- `--epochs 1` - Single epoch for quick testing
- `--batch 1` - Minimum batch size (use with grad accumulation)
- `--accum 4` - Gradient accumulation steps (effective batch = 1 × 4 = 4)
- `--workers 0` - No parallel data loading (avoids Windows multiprocessing issues)
- `--multi_scale 0` - Disable multi-scale training for faster testing

**Note:** The projector weights are excluded by default (`backbone.0.projector.*`) as they often mismatch when fine-tuning on new datasets.

#### Expected Output:
```
STAGE 1: DETECTOR TRAINING
================================================================================
Model Size:       nano
Resolution:       384
Epochs:           1
Device:           cuda
Dataset:          c:\Projects\RT-DETR-Seg\dataset
Output:           c:\Projects\RT-DETR-Seg\runs\detector_nano_384_1
...
```

#### Output Files:
```
runs/detector_nano_384_1/
├── best.pt              ← Best checkpoint (used for Stage 2)
├── last.pt              ← Last epoch checkpoint
├── checkpoint.pt        ← Latest checkpoint
├── metrics.json         ← Training metrics
└── tb/                  ← TensorBoard logs
```

---

### Stage 2: Train Segmentation Head (with Frozen Detector)

This loads the Stage 1 checkpoint and trains only the segmentation head while keeping the detector frozen.

```powershell
python train_seg_stage2.py `
    --size nano `
    --stage1_run runs/detector_nano_384_1 `
    --epochs 1 `
    --batch 1 `
    --accum 2 `
    --workers 0 `
    --seg_lr 0.0001 `
    --multi_scale 0 `
    --run_test 0
```

#### Parameters Explained:
- `--size nano` - Must match Stage 1 size
- `--stage1_run runs/detector_nano_384_1` - Path to Stage 1 output directory
- `--epochs 1` - Single epoch for quick testing
- `--batch 1` - Segmentation needs more memory, start small
- `--accum 2` - Lower accumulation (seg head is smaller than full model)
- `--seg_lr 0.0001` - Lower learning rate for fine-tuning segmentation head
- `--multi_scale 0` - Disable for faster testing
- `--run_test 0` - Skip test evaluation to save time

#### Expected Output:
```
STAGE 2: SEGMENTATION HEAD TRAINING
================================================================================
Model Size:       nano
Resolution:       384
Stage-1 Checkpoint: runs\detector_nano_384_1\best.pt
...
✓ Found Stage-1 checkpoint: runs\detector_nano_384_1\best.pt
✓ Validated: 10/10 sample annotations contain segmentation masks
...
✓ Detector frozen: 3456 parameters
✓ Segmentation head trainable: 789 parameters
...
```

#### Output Files:
```
runs/segmentation_nano_384_1/
├── best.pt              ← Best checkpoint (full model with segmentation)
├── last.pt              ← Last epoch checkpoint
├── stage2_config.json   ← Configuration used for Stage 2
├── metrics.json         ← Training metrics
└── tb/                  ← TensorBoard logs
```

---

## Validation Commands

### View Training Progress with TensorBoard

**Stage 1:**
```powershell
tensorboard --logdir runs/detector_nano_384_1/tb
```

**Stage 2:**
```powershell
tensorboard --logdir runs/segmentation_nano_384_1/tb
```

Then open: http://localhost:6006

### Verify Model Parameters

Check that detector is frozen in Stage 2:
```powershell
python -c "import torch; ckpt = torch.load('runs/segmentation_nano_384_1/stage2_config.json'); print(ckpt)"
```

---

## Troubleshooting

### Issue: "No checkpoint found in Stage-1 directory"
**Solution**: Ensure Stage 1 completed successfully and created `best.pt` or `last.pt`

### Issue: "No segmentation masks found in COCO annotations"
**Solution**: Your dataset needs instance segmentation annotations (polygons), not just bounding boxes. Stage 2 requires masks!

### Issue: "Out of memory (OOM)"
**Solutions**:
- Reduce `--batch` to 1
- Increase `--accum` to maintain effective batch size
- Use `--no_amp` to disable mixed precision (may help on some GPUs)
- Close other GPU-using applications

### Issue: "Frozen detector head out_dim mismatch"
**Solution**: The Stage 1 model must be trained on the same dataset (same number of classes) as Stage 2. Don't mix datasets between stages!

---

## Full Training Workflow

Once testing is complete, run full training:

### Stage 1 - Full Training (50+ epochs)
```powershell
python train_detector_stage1.py `
    --size nano `
    --epochs 50 `
    --batch 4 `
    --accum 4 `
    --workers 2 `
    --pretrain_exclude_keys "backbone.0.projector.*" `
    --multi_scale 1 `
    --expanded_scales 0
```

### Stage 2 - Full Training (50+ epochs)
```powershell
python train_seg_stage2.py `
    --size nano `
    --stage1_run runs/detector_nano_384_1 `
    --epochs 50 `
    --batch 2 `
    --accum 4 `
    --workers 2 `
    --seg_lr 0.0001 `
    --multi_scale 1 `
    --mask_loss_coef 1.0 `
    --dice_loss_coef 1.0 `
    --run_test 1
```

---

## Model Size Comparison

| Model  | Params | Resolution | Batch (4GB GPU) | Training Time (1 epoch) |
|--------|--------|------------|-----------------|-------------------------|
| Nano   | ~3M    | 384        | 4-8             | ~10 min                 |
| Small  | ~9M    | 512        | 2-4             | ~20 min                 |
| Medium | ~16M   | 576        | 1-2             | ~30 min                 |
| Base   | ~29M   | 560        | 1               | ~45 min                 |
| Large  | ~43M   | 560        | 1 (+ accum)     | ~60 min                 |

*Times are approximate and vary based on dataset size and GPU*

---

## Expected Training Curves

### Stage 1 (Detector)
- **Loss**: Should decrease from ~10-15 to ~2-5 (COCO)
- **mAP@0.5**: Should reach 30-50% (depends on dataset complexity)
- **Training time**: 10-30 min per epoch (nano model)

### Stage 2 (Segmentation)
- **Mask Loss**: Should decrease from ~0.5-1.0 to ~0.1-0.3
- **Dice Loss**: Should decrease from ~0.6-0.8 to ~0.2-0.4
- **Mask mAP**: Should reach 25-45% (typically 5-10% lower than bbox mAP)
- **Training time**: 15-40 min per epoch (nano model, depends on mask complexity)

---

## Next Steps

After successful testing:

1. **Increase epochs** to 50-100 for production models
2. **Enable multi-scale training** (`--multi_scale 1`) for better accuracy
3. **Tune hyperparameters**:
   - Learning rate (`--seg_lr` for Stage 2)
   - Batch size and accumulation
   - Loss coefficients (`--mask_loss_coef`, `--dice_loss_coef`)
4. **Monitor with TensorBoard** to track progress
5. **Use larger models** (small/base) for better accuracy if GPU allows

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `train_detector_stage1.py` | Stage 1: Train detector with bounding boxes |
| `train_seg_stage2.py` | Stage 2: Train segmentation head with frozen detector |
| `train_seg_single_stage.py` | Alternative: Train detector + segmentation end-to-end |
| `rfdetr/models/segmentation.py` | Segmentation head implementation |
| `rfdetr/detr.py` | Model wrappers (RFDETRNano, etc.) |
| `rfdetr/main.py` | Core training logic |

---

## Summary

✅ **Stage 1 Quick Test**:
```powershell
python train_detector_stage1.py --size nano --epochs 1 --batch 1 --accum 4 --workers 0 --multi_scale 0
```

✅ **Stage 2 Quick Test**:
```powershell
python train_seg_stage2.py --size nano --stage1_run runs/detector_nano_384_1 --epochs 1 --batch 1 --accum 2 --workers 0 --multi_scale 0 --run_test 0
```

Both commands should complete in **5-15 minutes** on a modest GPU (GTX 1650 Ti or better).
