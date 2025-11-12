# RT-DETR-Seg (RF-DETR fork)

Real-Time Detection Transformer with **Instance Segmentation** — a lightweight fork of RF‑DETR focused on fast training/inference and a simple two‑stage pipeline (detector ➜ masks).

> **License**: AGPL-3.0 — strong copyleft (including network use). See `LICENSE`.

---

## Highlights
- ⚡ **Real‑time friendly** model scales: `nano`, `small`, `base`, `medium`, `large`.
- 🎯 **Two‑stage training**:  
  1) Train detector (boxes + classes)  
  2) Freeze detector, train mask head (instance segmentation)
- 🧰 **Utilities**: `shrink_head.py`, `strip_class_heads.py`, ready-made training scripts, checkpoint tools.
- 📦 **Pretrained weights** provided (auto-download from hosted storage).
- 🧪 COCO‑style datasets (boxes + polygons/RLE for segmentation).

---

## Quick Start

### 1) Setup Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Validate setup
python validate_setup.py
```

**Linux/Mac:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio
pip install -r requirements.txt

# Validate setup
python validate_setup.py
```

### 2) Prepare Dataset

Use COCO-format JSON with this structure:
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

**Requirements:**
- **Stage 1 (Detection)**: Bounding boxes (`bbox` field)
- **Stage 2 (Segmentation)**: Bounding boxes + segmentation masks (`segmentation` field with polygons or RLE)

### 3) Train

#### **Stage 1 — Detection (Train Detector)**

```bash
python train_detector_stage1.py \
    --size nano \
    --epochs 50 \
    --batch 4 \
    --accum 4 \
    --workers 2 \
    --multi_scale 1
```

**Output:** `runs/detector_nano_384_1/checkpoint_best_regular.pth`

**Note:** The pretrained checkpoint has 91 classes (COCO), but your dataset likely has different classes. The projector weights are excluded by default to allow proper transfer learning.

#### **Prepare Checkpoint for Stage 2**

After Stage 1 completes, you need to shrink the detection head to match your dataset's classes:

```bash
python shrink_head.py \
    --in_ckpt runs/detector_nano_384_1/checkpoint_best_regular.pth \
    --out_ckpt runs/detector_nano_384_1/checkpoint_mydata.pth \
    --keep "class1" "class2" "class3"
```

**Example:**
```bash
# For a dataset with 3 classes: cells, blood_vessel, glomerulus
python shrink_head.py \
    --in_ckpt runs/detector_nano_384_1/checkpoint_best_regular.pth \
    --out_ckpt runs/detector_nano_384_1/checkpoint_3class.pth \
    --keep "cells" "blood_vessel" "glomerulus"
```

This creates a checkpoint with the correct number of output classes (your classes + background).

#### **Stage 2 — Segmentation (Train Mask Head)**

```bash
python train_seg_stage2.py \
    --size nano \
    --stage1_run runs/detector_nano_384_1 \
    --checkpoint runs/detector_nano_384_1/checkpoint_3class.pth \
    --epochs 50 \
    --batch 2 \
    --accum 4 \
    --workers 2 \
    --seg_lr 0.0001 \
    --multi_scale 1
```

**Output:** `runs/segmentation_nano_384_1/checkpoint_best_regular.pth` (full model with segmentation)

**What Happens:**
- Loads the shrunken Stage 1 checkpoint
- **Freezes detector** weights (537 params frozen)
- Trains **only segmentation head** (~26 trainable params for nano)
- Outputs instance segmentation masks

---

## Model Sizes

| Model  | Params | Resolution | Batch (4GB GPU) | Speed | Use Case |
|--------|--------|------------|-----------------|-------|----------|
| Nano   | ~3M    | 384        | 4-8             | ⚡⚡⚡ | Edge devices, real-time |
| Small  | ~9M    | 512        | 2-4             | ⚡⚡   | Balanced speed/accuracy |
| Medium | ~16M   | 576        | 1-2             | ⚡     | Higher accuracy |
| Base   | ~29M   | 560        | 1               | ⚡     | Production quality |
| Large  | ~43M   | 560        | 1 (+ accum)     | 🐢     | Maximum accuracy |

---

## Utilities

### `shrink_head.py`

Reduces the detection head output dimension to match your dataset's classes. **Required** after Stage 1 before running Stage 2.

```bash
python shrink_head.py \
    --in_ckpt <input_checkpoint.pth> \
    --out_ckpt <output_checkpoint.pth> \
    --keep "class1" "class2" "class3"
```

**When to use:**
- After Stage 1 training, before Stage 2
- When your dataset has fewer classes than the pretrained model
- To create a checkpoint compatible with Stage 2 frozen training

### `strip_class_heads.py`

Removes all classification head weights from a checkpoint. Useful for transfer learning when you want to completely retrain the classification layers.

```bash
python strip_class_heads.py <input_checkpoint.pth> <output_checkpoint.pth>
```

**When to use:**
- Transfer learning to completely different domains
- When class mismatch errors occur and you want to start fresh
- Creating backbone-only checkpoints

### `validate_setup.py`

Validates your environment, dependencies, and dataset before training.

```bash
python validate_setup.py
```

Checks:
- ✓ Python version (3.8+)
- ✓ PyTorch & CUDA availability
- ✓ Required dependencies
- ✓ Dataset structure and annotations
- ✓ Segmentation masks presence (for Stage 2)

---

## Training Tips

### Memory Management
- **OOM errors**: Reduce `--batch` to 1, increase `--accum` for gradient accumulation
- **4GB GPU**: Use nano/small models, batch=1, accum=4+, disable EMA
- **8GB+ GPU**: Medium/base models with batch=2-4

### Stage 2 Performance Tips
- **Batch size depends on GPU memory**:
  - **4GB GPU (GTX 1650 Ti)**: Must use `--batch 1 --accum 8`
  - **6GB GPU**: Can use `--batch 2 --accum 4` (2x faster than batch=1)
  - **8GB+ GPU**: Can use `--batch 4 --accum 2` (4x faster than batch=1)
- **Automatic optimization**: Nano models use lightweight `MaskHeadNano` (75% fewer parameters)
- **Expected speed**: With MaskHeadNano, Stage 2 takes similar time to Stage 1 per epoch
- **Workers**: Use `--workers 2` for parallel data loading (faster than `--workers 0`)

### Data Augmentation
- Start with `--multi_scale 0` for stability
- Enable `--multi_scale 1` after initial convergence
- Use `--expanded_scales 1` for more aggressive augmentation (large datasets only)

### Learning Rates
- **Stage 1 (Detector)**: Default LR (typically 5e-5 for nano)
- **Stage 2 (Segmentation)**: Lower LR (`--seg_lr 1e-4`) since detector is frozen

### Resolution Guidelines
- **384px**: Fast, good for large objects (nano default)
- **512px**: Balanced (small/medium default)
- **560-576px**: Better for small objects (base/large default)
- **Higher**: Increase if objects <32px in original images

### Early Stopping
- Enabled by default with patience=10
- Monitors validation mAP
- Stops if no improvement for N epochs

---

## Common Issues & Solutions

### Issue: "Stage 2 training extremely slow (8+ days)"

**Cause:** Large dataset with segmentation training (computationally expensive).

**Solution:** Use optimized settings based on your GPU:
```bash
# 4GB GPU (GTX 1650 Ti, RTX 3050 4GB)
python train_seg_stage2.py --size nano --batch 1 --accum 8 --workers 2 ...

# 6GB GPU (GTX 1660, RTX 3050 6GB) - 2x faster
python train_seg_stage2.py --size nano --batch 2 --accum 4 --workers 2 ...

# 8GB+ GPU (RTX 3060+) - 4x faster  
python train_seg_stage2.py --size nano --batch 4 --accum 2 --workers 2 ...
```

**Key optimizations:**
- Nano models automatically use `MaskHeadNano` (75% fewer parameters)
- Higher batch sizes dramatically reduce iterations (when GPU allows)
- With 4,950 images: batch=1 = 619 iterations, batch=2 = 310 iterations, batch=4 = 155 iterations

### Issue: "Frozen detector head out_dim mismatch"

**Cause:** Stage 1 checkpoint has different number of classes than your dataset.

**Solution:** Use `shrink_head.py` to match classes:
```bash
python shrink_head.py \
    --in_ckpt runs/detector_nano_384_1/checkpoint_best_regular.pth \
    --out_ckpt runs/detector_nano_384_1/checkpoint_mydata.pth \
    --keep "your" "class" "names"
```

### Issue: "No segmentation masks found"

**Cause:** COCO JSON missing `segmentation` field for Stage 2.

**Solution:** Ensure your annotations include polygon/RLE masks. Stage 2 requires instance segmentation annotations, not just bounding boxes.

### Issue: "size mismatch for backbone.0.projector"

**Cause:** Pretrained projector doesn't match your model configuration.

**Solution:** Already handled automatically - projector weights are excluded by default (`--pretrain_exclude_keys "backbone.0.projector.*"`).

### Issue: Out of memory (OOM)

**Solutions:**
1. Reduce batch size: `--batch 1`
2. Increase gradient accumulation: `--accum 8`
3. Disable EMA: Add `use_ema=False` in model config
4. Lower resolution: Try smaller model size
5. Reduce workers: `--workers 0`

---

## Monitoring Training

### TensorBoard

```bash
# Stage 1
tensorboard --logdir runs/detector_nano_384_1/tb

# Stage 2
tensorboard --logdir runs/segmentation_nano_384_1/tb
```

Open http://localhost:6006/

**Metrics to watch:**
- Total loss (should decrease steadily)
- mAP@50 (main detection metric)
- Mask loss & Dice loss (Stage 2)
- Learning rate schedule

### Weights & Biases (Optional)

Enable in training scripts with `--wandb` flag and set `WANDB_API_KEY` environment variable.

---

## Inference Example

```python
from rfdetr.detr import RFDETRNano
import cv2
import torch

# Load trained model
model = RFDETRNano(
    pretrain_weights="runs/segmentation_nano_384_1/checkpoint_best_regular.pth",
    num_classes=3,
    class_names=["cells", "blood_vessel", "glomerulus"],
    masks=True  # For segmentation
)

# Prepare image
image = cv2.imread("test.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run inference
with torch.no_grad():
    predictions = model.predict([image], threshold=0.5)

# predictions contains:
# - boxes: [N, 4] bounding boxes
# - scores: [N] confidence scores  
# - class_ids: [N] class indices
# - masks: [N, H, W] binary masks (if masks=True)
```

---

## File Structure

```
c:\Projects\RT-DETR-Seg/
├── rfdetr/                          # Core library
│   ├── models/                      # Model architectures
│   │   ├── segmentation.py         # DETRsegm wrapper
│   │   ├── backbone/               # Backbone (DINOv2 variants)
│   │   └── ...
│   ├── datasets/                    # Data loaders
│   ├── util/                        # Utilities
│   ├── detr.py                     # Model wrappers
│   ├── main.py                     # Core training logic
│   └── config.py                   # Model configurations
├── train_detector_stage1.py        # Stage 1 training script
├── train_seg_stage2.py             # Stage 2 training script
├── train_seg_single_stage.py       # End-to-end alternative
├── shrink_head.py                  # Checkpoint class reduction
├── strip_class_heads.py            # Remove classification heads
├── validate_setup.py               # Pre-training validation
├── requirements.txt                # Python dependencies
├── QUICK_START_GUIDE.md           # Detailed guide
└── runs/                          # Training outputs
    ├── detector_nano_384_1/       # Stage 1 results
    └── segmentation_nano_384_1/   # Stage 2 results
```

---

## License
This project is licensed under **GNU Affero General Public License v3.0 (AGPL‑3.0)**.  
If you modify and make the software available over a network (e.g., an API), you must publish your source changes under the same license.

---

## Citation
If you use this work, please cite:
```bibtex
@software{rtdetrseg_2025,
  title        = {RT-DETR-Seg: Real-Time Detection Transformer with Segmentation},
  author       = {Marko Mihajlovic},
  year         = {2025},
  url          = {https://github.com/mihajlov39547/rt-detr-seg}
}
```

## Acknowledgements
- Built upon DETR family and RF‑DETR ideas.
- Thanks to the open-source community for tools and baselines.
- Backbone: DINOv2 (Facebook AI Research)
- Original DETR: Facebook AI Research
