# GPU Memory Usage Guide for RT-DETR-Seg

This guide explains all parameters that affect GPU memory usage during training and provides optimized settings for different GPU configurations.

---

## 📊 Primary Memory Controls (Ordered by Impact)

### 1. **`--batch N`** 💥 CRITICAL - Linear Memory Scaling

**Impact Level:** ⚠️ **CRITICAL** - This is THE main parameter to control memory

- **`--batch 1`**: ~3.9GB used (works on 4GB GPU)
- **`--batch 2`**: ~7.8GB used (OOM on 4GB GPU!) ❌
- **`--batch 4`**: ~15.6GB used (needs 16GB+ GPU)

**What it does:** Controls how many images are processed simultaneously in a single forward/backward pass.

**Location:** All training scripts (`train_detector_stage1.py`, `train_seg_stage2.py`, `train_seg_single_stage.py`)

```bash
python train_seg_stage2.py --batch 1  # For 4GB GPU
python train_seg_stage2.py --batch 2  # For 6-8GB GPU
python train_seg_stage2.py --batch 4  # For 16GB+ GPU
```

---

### 2. **`--resolution N`** 💥 HIGH - Quadratic Memory Scaling

**Impact Level:** ⚠️ **HIGH** - Memory scales with resolution² (quadratic)

- **384×384**: 147,456 pixels per image (~3.9GB with batch=1)
- **512×512**: 262,144 pixels (1.78x more memory, ~6.5GB with batch=1) ❌
- **640×640**: 409,600 pixels (2.78x more memory, ~9GB with batch=1) ❌

**What it does:** Sets the input image resolution. Higher resolution = more feature map memory.

**Location:** Auto-set by `--size` unless explicitly overridden with `--resolution N`

**Default resolutions by model size:**
- `nano`: 384
- `small`: 512
- `base`: 560
- `medium`: 576
- `large`: 560

```bash
python train_seg_stage2.py --size nano --resolution 384  # Works on 4GB
python train_seg_stage2.py --size nano --resolution 320  # Even safer for 4GB
```

---

### 3. **`--size nano|small|base|medium|large`** ⚠️ MEDIUM - Model Size

**Impact Level:** ⚠️ **MEDIUM** - Affects model parameters + default resolution

| Model Size | Parameters | Default Resolution | Backbone         | Hidden Dim |
|------------|------------|-------------------|------------------|------------|
| nano       | 40.8M      | 384               | dinov2_small     | 256        |
| small      | ~41M       | 512               | dinov2_small     | 256        |
| base       | ~48M       | 560               | dinov2_base      | 256        |
| medium     | ~49M       | 576               | dinov2_small     | 256        |
| large      | ~68M       | 560               | dinov2_base      | 384        |

**What it does:** Determines model architecture, parameter count, and default resolution.

**Location:** All training scripts via `--size`

```bash
python train_seg_stage2.py --size nano   # Best for 4GB GPU
python train_seg_stage2.py --size small  # Needs 6-8GB GPU
python train_seg_stage2.py --size base   # Needs 16GB+ GPU
```

---

### 4. **`--accum N`** (Gradient Accumulation) ✅ NO MEMORY IMPACT!

**Impact Level:** ✅ **NONE** - Does NOT use extra GPU memory

- **`--accum 2`**: Process 1 image, accumulate gradients, process 1 more, then update weights
- **`--accum 4`**: Accumulate over 4 images before updating
- **`--accum 8`**: Accumulate over 8 images before updating

**What it does:** Simulates larger batch sizes without using more GPU memory. Processes images one-by-one (or in small batches) and accumulates gradients before updating model weights.

**Effective batch size = `batch × accum`**

**Example:**
- `--batch 1 --accum 8` → Effective batch = 8, but only uses memory for batch=1
- `--batch 2 --accum 4` → Effective batch = 8, uses memory for batch=2

**Location:** All training scripts via `--accum`

```bash
python train_seg_stage2.py --batch 1 --accum 8  # Simulates batch=8, uses memory for batch=1
```

**Trade-off:** More accumulation steps = slower training (more forward passes per weight update)

---

### 5. **`--multi_scale 0|1`** ⚠️ SMALL - Multi-Scale Training

**Impact Level:** ⚠️ **SMALL** - 5-10% memory overhead

- **`--multi_scale 0`**: Fixed resolution (recommended for 4GB GPU)
- **`--multi_scale 1`**: Trains on varying resolutions (better accuracy, more memory)

**What it does:** When enabled, randomly varies input resolution during training for better scale robustness.

**Location:** All training scripts via `--multi_scale`

```bash
python train_seg_stage2.py --multi_scale 0  # For 4GB GPU
python train_seg_stage2.py --multi_scale 1  # For 6GB+ GPU
```

---

### 6. **`--num_queries N`** ⚠️ SMALL - Object Queries

**Impact Level:** ⚠️ **SMALL** - Minimal impact

- **Default:** 300 queries (good balance)
- **Fewer queries** (100-200): Saves minimal memory, may hurt accuracy

**What it does:** Number of object queries in DETR architecture. Each query can detect one object.

**Location:** Most training scripts via `--num_queries`

**Recommendation:** Keep at default 300 unless you have extreme memory constraints.

---

### 7. **`--workers N`** ✅ CPU ONLY - No GPU Impact

**Impact Level:** ✅ **NONE** - Uses CPU RAM only

**What it does:** Number of CPU threads for data loading. Affects CPU memory and loading speed, not GPU memory.

**Recommendation:** 
- `workers=2`: Good default for most systems
- `workers=0`: Single-threaded, slower but safer
- `workers=4+`: Faster loading if you have many CPU cores

```bash
python train_seg_stage2.py --workers 2  # Recommended
```

---

## 🔧 Advanced Parameters (in Config Files)

### 8. **`amp`** (Automatic Mixed Precision) ✅ REDUCES MEMORY 30-40%

**Impact Level:** ✅ **REDUCES** - Saves 30-40% memory

**Status:** ✅ **Always enabled by default** (`amp=True` in `rfdetr/config.py`)

**What it does:** Uses float16 (half precision) instead of float32 for most operations, reducing memory usage significantly.

**Your GPU:** GTX 1650 Ti (pre-Ampere) uses **float16** automatically.

**Location:** `rfdetr/config.py` - `ModelConfig.amp = True`

⚠️ **Note:** This is why you can train at all on 4GB! Without AMP, you'd need ~6GB minimum.

---

### 9. **`gradient_checkpointing`** ⚠️ REDUCES MEMORY, SLOWER TRAINING

**Impact Level:** ⚠️ **REDUCES** - Trades compute for memory

**Status:** ❌ Disabled by default (`gradient_checkpointing=False`)

**What it does:** Saves memory by recomputing activations during backward pass instead of storing them. Reduces memory but increases training time (~20-30% slower).

**When to enable:** Only if you're running out of memory with batch=1

**Location:** `rfdetr/config.py` - `ModelConfig.gradient_checkpointing = False`

**Not needed** for your 4GB setup with batch=1 since AMP already saves enough memory.

---

## 📋 Memory Usage Examples (GTX 1650 Ti 4GB)

### ✅ WORKS (< 4GB)

| Command | Memory Used | Status |
|---------|-------------|--------|
| `--size nano --batch 1 --resolution 384` | ~3.9GB | ✅ Works |
| `--size nano --batch 1 --resolution 320` | ~3.2GB | ✅ Works |
| `--size nano --batch 1 --resolution 256` | ~2.5GB | ✅ Works (safer) |

### ❌ OOM (Out of Memory - > 4GB)

| Command | Memory Required | Status |
|---------|-----------------|--------|
| `--size nano --batch 2 --resolution 384` | ~7.8GB | ❌ OOM |
| `--size nano --batch 1 --resolution 512` | ~6.5GB | ❌ OOM |
| `--size small --batch 1 --resolution 512` | ~7.2GB | ❌ OOM |
| `--size base --batch 1 --resolution 560` | ~9.5GB | ❌ OOM |

---

## 🎯 Recommended Settings by GPU

### 4GB GPU (GTX 1650 Ti, GTX 1050 Ti, etc.)

**Detector Training (Stage 1):**
```bash
python train_detector_stage1.py \
    --size nano \
    --batch 2 \
    --accum 4 \
    --resolution 384 \
    --workers 2 \
    --multi_scale 0
```

**Segmentation Training (Stage 2):**
```bash
python train_seg_stage2.py \
    --size nano \
    --batch 1 \
    --accum 8 \
    --resolution 384 \
    --workers 2 \
    --multi_scale 0 \
    --stage1_run runs/detector_nano_384_1
```

**Single-Stage Training:**
```bash
python train_seg_single_stage.py \
    --size nano \
    --batch 1 \
    --accum 8 \
    --resolution 384 \
    --workers 2 \
    --multi_scale 0
```

---

### 6GB GPU (GTX 1660, RTX 2060, etc.)

**Detector Training:**
```bash
python train_detector_stage1.py \
    --size nano \
    --batch 4 \
    --accum 2 \
    --resolution 384 \
    --workers 2 \
    --multi_scale 1
```

**Segmentation Training:**
```bash
python train_seg_stage2.py \
    --size nano \
    --batch 2 \
    --accum 4 \
    --resolution 384 \
    --workers 2 \
    --multi_scale 1
```

---

### 8GB GPU (RTX 3060, RTX 2070, etc.)

**Detector Training:**
```bash
python train_detector_stage1.py \
    --size small \
    --batch 4 \
    --accum 2 \
    --resolution 512 \
    --workers 4 \
    --multi_scale 1
```

**Segmentation Training:**
```bash
python train_seg_stage2.py \
    --size small \
    --batch 2 \
    --accum 4 \
    --resolution 512 \
    --workers 4 \
    --multi_scale 1
```

---

### 16GB+ GPU (RTX 4090, A100, etc.)

**Detector Training:**
```bash
python train_detector_stage1.py \
    --size base \
    --batch 8 \
    --accum 1 \
    --resolution 560 \
    --workers 8 \
    --multi_scale 1
```

**Segmentation Training:**
```bash
python train_seg_stage2.py \
    --size base \
    --batch 4 \
    --accum 2 \
    --resolution 560 \
    --workers 8 \
    --multi_scale 1
```

---

## 💡 Key Takeaways

### For 4GB GPU (Your Setup):
1. ✅ **MUST use:** `--batch 1` with `--resolution 384` (or lower)
2. ✅ **Use:** `--accum 8` to simulate larger batch without memory cost
3. ✅ **Disable:** `--multi_scale 0` to save 5-10% memory
4. ❌ **Cannot use:** `batch=2` or `resolution > 384`
5. ✅ **Already enabled:** AMP (saves 30-40% memory automatically)

### Memory Scaling Summary:
- **Batch size:** Linear scaling (2x batch = 2x memory)
- **Resolution:** Quadratic scaling (2x resolution = 4x memory!)
- **Accumulation:** No memory impact (free larger effective batch)
- **Workers:** CPU only, no GPU impact
- **Model size:** Moderate impact (larger models = more parameters)

### Training Stage Comparison:
- **Stage 1 (Detector only):** Can use `batch=2` on 4GB GPU ✅
- **Stage 2 (Segmentation head):** Must use `batch=1` on 4GB GPU ⚠️
- **Single-stage (Full model):** Must use `batch=1` on 4GB GPU ⚠️

**Why?** Segmentation requires processing mask predictions for all 300 queries, which significantly increases memory usage compared to detection alone.

---

## 🐛 Troubleshooting OOM Errors

If you get `OutOfMemoryError: CUDA out of memory`, try these in order:

1. **Reduce batch size:** `--batch 2` → `--batch 1`
2. **Reduce resolution:** `--resolution 384` → `--resolution 320`
3. **Disable multi-scale:** `--multi_scale 1` → `--multi_scale 0`
4. **Reduce workers:** `--workers 2` → `--workers 0` (frees CPU RAM, may help)
5. **Reduce queries:** `--num_queries 300` → `--num_queries 200` (minimal help)
6. **Last resort:** Enable gradient checkpointing (see advanced config)

### Checking Current GPU Usage:

**Windows (PowerShell):**
```powershell
nvidia-smi
```

**Linux:**
```bash
watch -n 1 nvidia-smi
```

---

## 📁 Where to Find Parameters

### Training Scripts:
- `train_detector_stage1.py` - Stage 1 detector training
- `train_seg_stage2.py` - Stage 2 segmentation head training
- `train_seg_single_stage.py` - Single-stage end-to-end training

### Configuration Files:
- `rfdetr/config.py` - Model and training configuration classes
- `rfdetr/main.py` - Core training loop and argument parsing

### Key Config Classes:
- `ModelConfig` - Model architecture settings (amp, gradient_checkpointing, resolution)
- `TrainConfig` - Training hyperparameters (batch_size, grad_accum_steps, lr)
- `RFDETRNanoConfig` - Nano model defaults
- `RFDETRSmallConfig` - Small model defaults
- `RFDETRBaseConfig` - Base model defaults

---

## 📊 Quick Reference Table

| Parameter | Memory Impact | Your 4GB GPU | 6GB GPU | 8GB+ GPU |
|-----------|---------------|--------------|---------|----------|
| `--batch` | 💥 CRITICAL | 1 | 2 | 2-4 |
| `--resolution` | 💥 HIGH | 384 | 384-512 | 512+ |
| `--size` | ⚠️ MEDIUM | nano | nano/small | small/base |
| `--accum` | ✅ NONE | 8 | 4 | 2-4 |
| `--multi_scale` | ⚠️ SMALL | 0 | 1 | 1 |
| `--num_queries` | ⚠️ SMALL | 300 | 300 | 300 |
| `--workers` | ✅ CPU only | 2 | 2-4 | 4-8 |

---

## 🎓 Understanding Effective Batch Size

**Effective Batch Size = `batch × accum`**

This is the number of images processed before updating model weights.

### Examples:

| Batch | Accum | Effective Batch | Memory Used | Speed |
|-------|-------|-----------------|-------------|-------|
| 1 | 8 | 8 | 1x (3.9GB) | Slower |
| 2 | 4 | 8 | 2x (7.8GB) ❌ | Faster |
| 4 | 2 | 8 | 4x (15.6GB) ❌ | Fastest |

**All three configurations achieve the same effective batch size (8), but only the first works on 4GB GPU!**

### Why use accumulation?
- Maintains training quality with limited GPU memory
- Larger effective batches → better gradient estimates → better convergence
- Trade-off: Slower training (more forward passes per weight update)

---

## 📖 Additional Resources

- **QUICK_START_GUIDE.md** - Step-by-step training workflow
- **README.md** - Project overview and setup
- **nvidia-smi** - Monitor real-time GPU usage
- **TensorBoard** - Monitor training metrics (use `--tensorboard` flag)

---

**Last Updated:** November 12, 2025  
**Hardware:** NVIDIA GeForce GTX 1650 Ti (4GB VRAM), CUDA 12.1
