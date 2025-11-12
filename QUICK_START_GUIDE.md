# Quick Start Guide - RF-DETR-Seg Two-Stage Training

This guide shows you how to train an instance segmentation model using the two-stage approach: first train a detector, then add segmentation while keeping the detector frozen.

## 🚀 TL;DR - Quick Commands

**Validate Setup:**
```powershell
python validate_setup.py
```

**Stage 1 - Train Detector (1 epoch test):**
```powershell
python train_detector_stage1.py --size nano --epochs 1 --batch 1 --accum 4 --workers 0 --multi_scale 0
```

**Shrink Checkpoint (match your dataset classes):**
```powershell
python shrink_head.py --in_ckpt runs/detector_nano_384_1/checkpoint_best_regular.pth --out_ckpt runs/detector_nano_384_1/checkpoint_mydata.pth --keep "class1" "class2" "class3"
```

**Stage 2 - Train Segmentation (1 epoch test):**
```powershell
python train_seg_stage2.py --size nano --stage1_run runs/detector_nano_384_1 --checkpoint runs/detector_nano_384_1/checkpoint_mydata.pth --epochs 1 --batch 4 --accum 2 --workers 2 --multi_scale 0 --run_test 0
```

**💡 Tip:** Use `--batch 4` for Stage 2 to get **10-15x faster training** than batch=1! If OOM, reduce to `--batch 2`.

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

### ⚠️ IMPORTANT: Shrink Checkpoint Before Stage 2

If you loaded pretrained weights in Stage 1 (which happens by default), the checkpoint will have **91 output classes** from COCO pretraining. Your dataset likely has fewer classes, causing a dimension mismatch error in Stage 2.

**You MUST shrink the checkpoint to match your dataset classes:**

#### Step 1: Check Your Dataset Classes
```powershell
python -c "import json; data=json.load(open('dataset/train/_annotations.coco.json')); print('Categories:', [c['name'] for c in data['categories']])"
```

Example output:
```
Categories: ['cells', 'blood_vessel', 'glomerulus']
```

#### Step 2: Shrink the Checkpoint
```powershell
python shrink_head.py `
    --in_ckpt runs/detector_nano_384_1/checkpoint_best_regular.pth `
    --out_ckpt runs/detector_nano_384_1/checkpoint_mydata.pth `
    --keep "cells" "blood_vessel" "glomerulus"
```

Replace `"cells" "blood_vessel" "glomerulus"` with YOUR actual class names from Step 1.

**What this does:**
- Reduces the detection head from 91 output dimensions (COCO classes) to N+1 dimensions (your classes + background)
- Preserves weights for the classes you specify (by name matching with COCO classes)
- Keeps the background class
- Creates a new checkpoint compatible with Stage 2

#### Step 3: Verify the Shrink
```powershell
python -c "import torch; ckpt=torch.load('runs/detector_nano_384_1/checkpoint_mydata.pth'); print(f\"Output dim: {ckpt['model']['class_embed.0.weight'].shape[0]}\")"
```

Expected output for 3 classes:
```
Output dim: 4
```
(3 foreground classes + 1 background = 4 total)

---

### Stage 2: Train Segmentation Head (with Frozen Detector)

This loads the **shrunken** Stage 1 checkpoint and trains only the segmentation head while keeping the detector frozen.

```powershell
python train_seg_stage2.py `
    --size nano `
    --stage1_run runs/detector_nano_384_1 `
    --checkpoint runs/detector_nano_384_1/checkpoint_mydata.pth `
    --epochs 1 `
    --batch 4 `
    --accum 2 `
    --workers 2 `
    --seg_lr 0.0001 `
    --multi_scale 0 `
    --run_test 0
```

#### Parameters Explained:
- `--size nano` - Must match Stage 1 size
- `--stage1_run runs/detector_nano_384_1` - Path to Stage 1 output directory
- `--checkpoint runs/detector_nano_384_1/checkpoint_mydata.pth` - **Use the shrunken checkpoint from shrink_head.py**
- `--epochs 1` - Single epoch for quick testing
- `--batch 4` - **Higher batch size for faster training** (10-15x speedup vs batch=1!)
- `--accum 2` - Gradient accumulation (effective batch = 4 × 2 = 8)
- `--workers 2` - Parallel data loading for speed
- `--seg_lr 0.0001` - Lower learning rate for fine-tuning segmentation head
- `--multi_scale 0` - Disable for faster testing
- `--run_test 0` - Skip test evaluation to save time

**💡 Performance Tip:** Stage 2 uses optimized `MaskHeadNano` for nano models (75% fewer parameters). Combined with batch=4, this gives **10-15x faster training** compared to batch=1!

**If OOM (Out of Memory):**
```powershell
# Reduce batch to 2 (still 2x faster than batch=1)
python train_seg_stage2.py ... --batch 2 --accum 4 ...
```

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

### Issue: "Stage 2 training extremely slow (days for 1 epoch)"
**Cause**: Batch size too small (batch=1) with large dataset.

**Solution**: Increase batch size for massive speedup:
```powershell
# Try batch=4 first (10-15x faster!)
--batch 4 --accum 2 --workers 2

# If OOM, use batch=2 (still 2x faster)
--batch 2 --accum 4 --workers 2
```

**Why this works:** Nano models use optimized `MaskHeadNano` with 75% fewer parameters. Higher batch sizes process more images in parallel, dramatically reducing training time.

### Issue: "Frozen detector head out_dim (91) != requested (4)"
**Cause**: Stage 1 checkpoint has 91 classes (COCO pretrained) but your dataset has fewer classes.

**Solution**: Use `shrink_head.py` to reduce the checkpoint dimensions:
```powershell
# Check your dataset classes first
python -c "import json; data=json.load(open('dataset/train/_annotations.coco.json')); print('Categories:', [c['name'] for c in data['categories']])"

# Shrink the checkpoint to match
python shrink_head.py --in_ckpt runs/detector_nano_384_1/checkpoint_best_regular.pth --out_ckpt runs/detector_nano_384_1/checkpoint_mydata.pth --keep "class1" "class2" "class3"

# Then use the shrunken checkpoint in Stage 2
python train_seg_stage2.py --checkpoint runs/detector_nano_384_1/checkpoint_mydata.pth ...
```

### Issue: "No checkpoint found in Stage-1 directory"
**Solution**: Ensure Stage 1 completed successfully and created `best.pt` or `last.pt`

### Issue: "No segmentation masks found in COCO annotations"
**Solution**: Your dataset needs instance segmentation annotations (polygons), not just bounding boxes. Stage 2 requires masks!

### Issue: "Out of memory (OOM)"
**Solutions**:
- **For Stage 2**: Reduce `--batch` from 4 to 2 or 1
- Increase `--accum` to maintain effective batch size
- Reduce `--workers` to 0 (disables parallel loading but saves memory)
- Use `--no_amp` to disable mixed precision (may help on some GPUs)
- Close other GPU-using applications

**Note:** Even batch=2 is much faster than batch=1 for Stage 2!

---

## Full Training Workflow

Once testing is complete, run full training with the complete workflow:

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

### Shrink Checkpoint After Stage 1
```powershell
# Check your dataset classes
python -c "import json; data=json.load(open('dataset/train/_annotations.coco.json')); print('Categories:', [c['name'] for c in data['categories']])"

# Shrink checkpoint to match dataset
python shrink_head.py `
    --in_ckpt runs/detector_nano_384_1/checkpoint_best_regular.pth `
    --out_ckpt runs/detector_nano_384_1/checkpoint_mydata.pth `
    --keep "class1" "class2" "class3"  # Replace with YOUR class names
```

### Stage 2 - Full Training (50+ epochs)
```powershell
python train_seg_stage2.py `
    --size nano `
    --stage1_run runs/detector_nano_384_1 `
    --checkpoint runs/detector_nano_384_1/checkpoint_mydata.pth `
    --epochs 50 `
    --batch 4 `
    --accum 2 `
    --workers 2 `
    --seg_lr 0.0001 `
    --multi_scale 1 `
    --mask_loss_coef 1.0 `
    --dice_loss_coef 1.0 `
    --run_test 1
```

**💡 Optimization Note:** Nano models automatically use lightweight `MaskHeadNano` (75% fewer parameters). Use `--batch 4` for best performance (10-15x faster than batch=1).

---

## Model Size Comparison

| Model  | Params | Resolution | Batch (4GB GPU) | Stage 1 Time | Stage 2 Time* |
|--------|--------|------------|-----------------|--------------|---------------|
| Nano   | ~3M    | 384        | 4-8             | ~10 min      | ~10-15 min    |
| Small  | ~9M    | 512        | 2-4             | ~20 min      | ~20-25 min    |
| Medium | ~16M   | 576        | 1-2             | ~30 min      | ~30-40 min    |
| Base   | ~29M   | 560        | 1               | ~45 min      | ~50-60 min    |
| Large  | ~43M   | 560        | 1 (+ accum)     | ~60 min      | ~70-80 min    |

*Times are approximate per epoch. Stage 2 times assume proper batch size (not batch=1).

**Important:** Stage 2 training time depends heavily on batch size. With batch=1, Stage 2 can be 10-15x slower!

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

✅ **Complete Two-Stage Workflow**:

**1. Train Detector (Stage 1):**
```powershell
python train_detector_stage1.py --size nano --epochs 1 --batch 1 --accum 4 --workers 0 --multi_scale 0
```

**2. Shrink Checkpoint to Match Dataset:**
```powershell
# Check your classes
python -c "import json; data=json.load(open('dataset/train/_annotations.coco.json')); print('Categories:', [c['name'] for c in data['categories']])"

# Shrink the checkpoint
python shrink_head.py --in_ckpt runs/detector_nano_384_1/checkpoint_best_regular.pth --out_ckpt runs/detector_nano_384_1/checkpoint_mydata.pth --keep "class1" "class2" "class3"
```

**3. Train Segmentation Head (Stage 2 - OPTIMIZED):**
```powershell
python train_seg_stage2.py --size nano --stage1_run runs/detector_nano_384_1 --checkpoint runs/detector_nano_384_1/checkpoint_mydata.pth --epochs 1 --batch 4 --accum 2 --workers 2 --multi_scale 0 --run_test 0
```

All three steps should complete in **15-20 minutes** on a modest GPU (GTX 1650 Ti or better).

**⚠️ Critical Tips:**
- **Don't skip shrink_head.py!** Without it, you'll get a dimension mismatch error in Stage 2
- **Use batch=4 for Stage 2!** This gives 10-15x speedup compared to batch=1
- **Nano models automatically use MaskHeadNano** (75% fewer parameters, much faster training)
