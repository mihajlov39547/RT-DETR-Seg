# RT-DETR-Seg (RF-DETR fork)

Real-Time Detection Transformer with **Instance Segmentation** — a lightweight fork of RF‑DETR focused on fast training/inference and a simple two‑stage pipeline (detector ➜ masks).

> **License**: AGPL-3.0 — strong copyleft (including network use). See `LICENSE`.

---

## Highlights
- ⚡ **Real‑time friendly** model scales: `nano`, `small`, `base`, `medium`, `large`.
- 🎯 **Two‑stage training**:  
  1) Train detector (boxes + classes)  
  2) Freeze detector, train mask head (instance segmentation)
- 🧰 **Utilities**: `shrink_head.py`, ready-made training scripts, checkpoint tools.
- 📦 **Pretrained weights** provided (tracked with Git LFS).
- 🧪 COCO‑style datasets (boxes + polygons/RLE).

---

## Quickstart

### 1) Install
```bash
git clone https://github.com/mihajlov39547/rt-detr-seg.git
cd rt-detr-seg
python -m venv detr-env && source detr-env/bin/activate   # on Windows: detr-env\Scripts\activate
pip install -U pip
pip install -e .
```

> If you use the included `.pth` checkpoints, enable **Git LFS** first:
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes && git commit -m "track weights with LFS"
```

### 2) Data
Use COCO-style JSON (instances). Update dataset paths in your config/args.
Common layout:
```
datasets/
  your_dataset/
    annotations/
      instances_train.json
      instances_val.json
    images/
      train/
      val/
```

### 3) Train

**Stage 1 — Detection**
```bash
python train_detector_stage1.py   --size small   --data datasets/your_dataset   --epochs 300   --output_dir runs/det_small   --imgsz 512   --batch 8
```

**Stage 2 — Segmentation**
```bash
python train_seg_stage2.py   --size small   --stage1_ckpt runs/det_small/checkpoint_best.pth   --data datasets/your_dataset   --epochs 150   --output_dir runs/seg_small   --imgsz 512   --batch 4
```

### 4) Inference (example)
```python
from rfdetr import load_model
import torch, cv2

model = load_model("rf-detr-small.pth", task="seg")  # or "det"
img = cv2.imread("demo.jpg")[:, :, ::-1]  # BGR->RGB
with torch.no_grad():
    outputs = model.predict([img], conf=0.25)
# outputs: boxes / masks / scores / classes
```

---

## Repo Structure
```
rfdetr/                 # library code
train_detector_stage1.py
train_seg_stage2.py
shrink_head.py
runs/                   # logs & checkpoints
docs/                   # docs & examples
```

---

## Tips
- **Resolution**: 512×512 is a good starting point; increase if objects are small.
- **Augmentations**: start conservative for stability; ramp up after first baseline.
- **GPU Memory**: lower `--imgsz` or `--batch` if you hit OOM.

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
