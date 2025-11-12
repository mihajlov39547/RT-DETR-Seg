# train_seg_single_stage.py
"""
Single-stage instance segmentation training for RF-DETR
(backbone + transformer + detection head + mask head together).

- Starts from COCO-pretrained RF-DETR weights (or your own checkpoint).
- Enables masks and aux losses.
- Does NOT freeze the detector (whole model trains end-to-end).
- Uses your existing RF-DETR wrappers & TrainConfig API.

Notes:
- Keep class set/order consistent with your dataset JSON.
- If you pass a detector checkpoint via --init_ckpt, it will be loaded
  without freezing; head will be re-initialized if needed by RF-DETR logic.
- Nano models automatically use optimized MaskHeadNano (75% fewer parameters).
- Requires more GPU memory than two-stage training (detector not frozen).

Performance Tips:
- 4GB GPU: Use nano with --batch 1 --accum 8
- 6GB GPU: Can use small with --batch 2 --accum 4
- 8GB+ GPU: Can use small/medium with --batch 4 --accum 2

Usage Examples:

Quick test (1 epoch, 4GB GPU):
    python train_seg_single_stage.py --size nano --epochs 1 --batch 1 --accum 8 --workers 2 --multi_scale 0

Full training (50 epochs, 4GB GPU):
    python train_seg_single_stage.py --size nano --epochs 50 --batch 1 --accum 8 --workers 2 --tensorboard

With 6GB+ GPU (faster):
    python train_seg_single_stage.py --size nano --epochs 50 --batch 2 --accum 4 --workers 2 --tensorboard

Small model (8GB+ GPU):
    python train_seg_single_stage.py --size small --epochs 50 --batch 2 --accum 4 --workers 2 --tensorboard
"""

import argparse, json, os, sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

# Safer, faster matmul + cudnn autotune
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
cudnn.benchmark = True

# Make repo importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

# Resolution defaults per model size
SIZE_DEFAULTS = {
    "nano":   384,
    "small":  512,
    "base":   560,
    "medium": 576,
    "large":  560,
}

DATASET_DIR = REPO_ROOT / "dataset"  # expects train/ valid/ test/ each with _annotations.coco.json

def read_coco_categories(ann_path: str):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data.get("categories", []), key=lambda c: c.get("id", 0))
    names = [c["name"] for c in cats]
    return names, len(names)

def parse_args():
    p = argparse.ArgumentParser("RF-DETR Single-Stage Segmentation Trainer")
    # Core
    p.add_argument("--size", choices=list(SIZES.keys()), default="nano",
                   help="Model size (nano recommended for 4GB GPU)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: runs/seg_single_<size>_<res>_N)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resolution", type=int, default=None,
                   help="Image resolution (default: size-specific)")
    # Init weights
    p.add_argument("--pretrain_weights", type=str, default=None,
                   help="COCO RF-DETR weights filename or path (auto-download if known name, default: rf-detr-<size>.pth)")
    p.add_argument("--pretrain_exclude_keys", type=str, default="backbone.0.projector.*",
                   help="Keys to exclude when loading pretrain weights (default: projector)")
    p.add_argument("--init_ckpt", type=str, default="",
                   help="Optional: path to your own detector/seg checkpoint to initialize from (NOT frozen).")
    # Train
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of epochs to train")
    p.add_argument("--batch", type=int, default=1,
                   help="Batch size per step (1 for 4GB GPU, 2 for 6GB, 4 for 8GB+)")
    p.add_argument("--accum", type=int, default=8,
                   help="Gradient accumulation steps (8 for batch=1, 4 for batch=2, 2 for batch=4)")
    p.add_argument("--workers", type=int, default=2,
                   help="Number of dataloader workers")
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Learning rate (lower for single-stage with all trainable)")
    p.add_argument("--lr_encoder", type=float, default=1e-4,
                   help="Encoder learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    # DETR knobs
    p.add_argument("--num_queries", type=int, default=300,
                   help="Number of object queries (default: 300)")
    p.add_argument("--multi_scale", type=int, default=1,
                   help="Use multi-scale training (1=enabled, 0=disabled)")
    p.add_argument("--expanded_scales", type=int, default=0,
                   help="Use expanded scale set (0=disabled, 1=enabled)")
    # Loss weights
    p.add_argument("--mask_loss_coef", type=float, default=1.0)
    p.add_argument("--dice_loss_coef", type=float, default=1.0)
    p.add_argument("--cls_loss_coef", type=float, default=1.0)
    p.add_argument("--bbox_loss_coef", type=float, default=5.0)
    p.add_argument("--giou_loss_coef", type=float, default=2.0)
    # Scheduler
    p.add_argument("--lr_scheduler", type=str, default="cosine",
                   help="Learning rate scheduler (cosine or step)")
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Number of warmup epochs")
    # Early stopping
    p.add_argument("--early_stopping", type=int, default=1,
                   help="Enable early stopping (1=enabled, 0=disabled)")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience")
    # Logging
    p.add_argument("--tensorboard", action="store_true",
                   help="Enable TensorBoard logging")
    p.add_argument("--run_test", action="store_true",
                   help="Run test evaluation after training")
    return p.parse_args()

def main():
    args = parse_args()
    
    # Set defaults based on model size
    size = args.size
    resolution = args.resolution if args.resolution else SIZE_DEFAULTS[size]
    pretrain_weights = args.pretrain_weights if args.pretrain_weights else f"rf-detr-{size}.pth"
    
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        run_dir = REPO_ROOT / "runs"
        run_dir.mkdir(exist_ok=True)
        # Find next available run number
        existing = [d.name for d in run_dir.iterdir() if d.is_dir() and d.name.startswith(f"seg_single_{size}_{resolution}_")]
        run_num = 1
        while f"seg_single_{size}_{resolution}_{run_num}" in existing:
            run_num += 1
        args.output_dir = str(run_dir / f"seg_single_{size}_{resolution}_{run_num}")
    
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

    # Display configuration
    print("="*80)
    print("SINGLE-STAGE INSTANCE SEGMENTATION TRAINING")
    print("="*80)
    print(f"Model Size:       {size}")
    print(f"Resolution:       {resolution}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Device:           {args.device}")
    print(f"Dataset:          {DATASET_DIR}")
    print(f"Pretrain Weights: {pretrain_weights}")
    print(f"Init Checkpoint:  {args.init_ckpt if args.init_ckpt else 'None'}")
    print(f"Classes ({num_classes}): {class_names}")
    print("="*80)
    print()

    # Build base model (masks=True, NOT frozen)
    ModelCls = SIZES[size]
    # pick init: if init_ckpt provided, use it; else COCO pretrain
    init_path = args.init_ckpt if args.init_ckpt else pretrain_weights

    print("Initializing model (unfrozen, full end-to-end training)...")
    model = ModelCls(
        resolution=resolution,
        model_size=size,  # Pass size for MaskHeadNano selection
        # train whole model (not frozen)
        frozen_weights=None,
        device=args.device,
        num_classes=num_classes,
        class_names=class_names,
        masks=True,            # enable segmentation head/wrapper
        aux_loss=True,         # critical for DETR-style training
        num_queries=args.num_queries,
        pretrain_weights=init_path,   # will auto-download if using known RF-DETR filename
        pretrain_exclude_keys=args.pretrain_exclude_keys,
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
        square_resize_div_64=(size in {"nano", "small", "medium"}),
        # data/augs
        multi_scale=bool(args.multi_scale),
        expanded_scales=bool(args.expanded_scales),
        do_random_resize_via_padding=False,
        # scheduler/ES
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        early_stopping=bool(args.early_stopping),
        early_stopping_patience=args.patience,
        early_stopping_min_delta=0.001,
        early_stopping_use_ema=False,
        # misc
        seed=args.seed,
        use_ema=False  # Disable EMA to save memory
    )

    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Epochs:            {args.epochs}")
    print(f"Batch size:        {args.batch}")
    print(f"Grad accumulation: {args.accum}")
    print(f"Effective batch:   {args.batch * args.accum}")
    print(f"Learning rate:     {args.lr}")
    print(f"Encoder LR:        {args.lr_encoder}")
    print(f"Workers:           {args.workers}")
    print(f"Multi-scale:       {bool(args.multi_scale)}")
    print(f"LR Scheduler:      {args.lr_scheduler}")
    print(f"Warmup epochs:     {args.warmup_epochs}")
    print(f"Early stopping:    {bool(args.early_stopping)} (patience={args.patience})")
    print("="*80)
    print()

    # GPU memory tip
    if size == "nano":
        print("💡 Tip: Nano model uses optimized MaskHeadNano (75% fewer parameters)")
    if args.batch == 1:
        print("💡 Tip: Using batch=1 for 4GB GPU. Consider batch=2 for 6GB+ or batch=4 for 8GB+")
    print()

    # Train end-to-end (this uses your rfdetr.main train loop)
    print("Starting single-stage training (detector + segmentation head unfrozen)...")
    model.train_from_config(tcfg)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Artifacts saved to: {args.output_dir}")
    print()

if __name__ == "__main__":
    main()
