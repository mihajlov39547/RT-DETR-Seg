# train_seg_single_stage.py
"""
Single-stage instance segmentation training for RF-DETR
(backbone + transformer + detection head + mask head together).

- Starts from COCO-pretrained RF-DETR weights (or your own checkpoint).
- Enables masks and aux losses.
- Does NOT freeze the detector (whole model trains).
- Uses your existing RF-DETR wrappers & TrainConfig API.

Notes:
- Keep class set/order consistent with your dataset JSON.
- If you pass a detector checkpoint via --init_ckpt, it will be loaded
  without freezing; head will be re-initialized if needed by RF-DETR logic.
"""

import argparse, json, os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

# Safer, faster matmul + cudnn autotune
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
cudnn.benchmark = True

# RF-DETR wrappers
from rfdetr.detr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge
from rfdetr.config import TrainConfig

SIZES = {
    "nano":   RFDETRNano,
    "small":  RFDETRSmall,
    "base":   RFDETRBase,
    "medium": RFDETRMedium,
    "large":  RFDETRLarge,
}

REPO_ROOT = Path(r"/content/drive/MyDrive/RT-DETR-Seg").resolve()
DATASET_DIR = REPO_ROOT / "dataset"  # expects train/ val/ test/ each with _annotations.coco.json

def read_coco_categories(ann_path: str):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data.get("categories", []), key=lambda c: c.get("id", 0))
    names = [c["name"] for c in cats]
    return names, len(names)

def parse_args():
    p = argparse.ArgumentParser("RF-DETR Single-Stage Segmentation Trainer")
    # Core
    p.add_argument("--size", choices=list(SIZES.keys()), default="small")
    p.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "runs" / "seg_single_stage"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resolution", type=int, default=512)
    # Init weights
    p.add_argument("--pretrain_weights", type=str, default="rf-detr-small.pth",
                   help="COCO RF-DETR weights filename or path (auto-download if known name)")
    p.add_argument("--init_ckpt", type=str, default="",
                   help="Optional: path to your own detector/seg checkpoint to initialize from (NOT frozen).")
    # Train
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_encoder", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    # DETR knobs
    p.add_argument("--num_queries", type=int, default=150)
    p.add_argument("--multi_scale", type=int, default=0)
    p.add_argument("--expanded_scales", type=int, default=0)
    # Mask head perf/VRAM
    p.add_argument("--mask_chunk", type=int, default=32, help="Chunk size for mask head forward")
    # Loss weights
    p.add_argument("--mask_loss_coef", type=float, default=2.0)
    p.add_argument("--dice_loss_coef", type=float, default=2.0)
    p.add_argument("--cls_loss_coef", type=float, default=1.0)
    p.add_argument("--bbox_loss_coef", type=float, default=5.0)
    p.add_argument("--giou_loss_coef", type=float, default=2.0)
    # Logging
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--run_test", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_ann = DATASET_DIR / "train" / "_annotations.coco.json"
    if not train_ann.exists():
        raise FileNotFoundError(f"Missing: {train_ann}")

    # Require instance masks present
    with open(train_ann, "r", encoding="utf-8") as f:
        anns = json.load(f)
    if not any(a.get("segmentation") for a in anns.get("annotations", [])):
        raise ValueError("Dataset must include COCO-style instance masks (segmentation).")

    class_names, num_classes = read_coco_categories(str(train_ann))

    # Build base model (masks=True, NOT frozen)
    ModelCls = SIZES[args.size]
    # pick init: if init_ckpt provided, use it; else COCO pretrain
    init_path = args.init_ckpt if args.init_ckpt else args.pretrain_weights

    model = ModelCls(
        resolution=args.resolution,
        # train whole model (not frozen)
        frozen_weights=None,
        device=args.device,
        num_classes=num_classes,
        class_names=class_names,
        masks=True,            # enable segmentation head/wrapper
        aux_loss=True,         # critical for DETR-style training
        num_queries=args.num_queries,
        pretrain_weights=init_path,   # will auto-download if using known RF-DETR filename
        multi_scale=bool(args.multi_scale),
        expanded_scales=bool(args.expanded_scales),
    )

    # TrainConfig for single-stage
    tcfg = TrainConfig(
        dataset_dir=str(DATASET_DIR),
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        num_workers=args.workers,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        tensorboard=args.tensorboard,
        run_test=args.run_test,
        # losses
        cls_loss_coef=args.cls_loss_coef,
        bbox_loss_coef=args.bbox_loss_coef,
        giou_loss_coef=args.giou_loss_coef,
        mask_loss_coef=args.mask_loss_coef,
        dice_loss_coef=args.dice_loss_coef,
        # important flags
        masks=True,
        aux_loss=True,
        frozen_weights=None,     # NOT freezing
        class_names=class_names,
        square_resize_div_64=True,
        # data/augs
        multi_scale=bool(args.multi_scale),
        expanded_scales=bool(args.expanded_scales),
        do_random_resize_via_padding=False,
        # scheduler/ES
        lr_scheduler='step',
        warmup_epochs=0,
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_use_ema=False,
        # misc
        seed=args.seed,
    )

    print(f"==> Single-stage seg training (unfrozen) on {args.device} | size={args.size} | res={args.resolution}")
    print(f"    dataset_dir={DATASET_DIR}")
    print(f"    init_ckpt={init_path}")
    print(f"    classes({num_classes})={class_names}")

    # Train end-to-end (this uses your rfdetr.main train loop)
    model.train_from_config(tcfg)

    print("\nSingle-stage training complete.")
    print(f"Artifacts in: {args.output_dir}")

if __name__ == "__main__":
    main()
