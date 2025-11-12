"""
train_detector_stage1.py

Usage:
python train_detector_stage1.py --size nano --epochs 1 --batch 1 --accum 4 --workers 0 --pretrain_exclude_keys "backbone.0.projector.*" --multi_scale 0 --expanded_scales 0
python train_detector_stage1.py --size base --epochs 1 --batch 1 --accum 4 --workers 0 --pretrain_exclude_keys "backbone.0.projector.*" --multi_scale 0 --expanded_scales 0
"""
import sys, argparse
from pathlib import Path
import torch
from itertools import count
from torch.utils.tensorboard import SummaryWriter

# Safe matmul precision (no-op if unsupported)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Make repo importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import RF-DETR model wrappers
from rfdetr.detr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge

SIZE2CLS = {
    "nano":   RFDETRNano,
    "small":  RFDETRSmall,
    "base":   RFDETRBase,
    "medium": RFDETRMedium,
    "large":  RFDETRLarge,
}

# Handy defaults per size (only things you might want to override here)
SIZE_DEFAULTS = {
    "nano":   {"resolution": 384, "pretrain_weights": "rf-detr-nano.pth"},
    "small":  {"resolution": 512, "pretrain_weights": "rf-detr-small.pth"},
    "base":  {"resolution": 560, "pretrain_weights": "rf-detr-base.pth"},
    "medium": {"resolution": 576, "pretrain_weights": "rf-detr-medium.pth"},
    "large":  {"resolution": 560, "pretrain_weights": "rf-detr-large.pth"},
}

import json

def read_coco_categories(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data.get("categories", []), key=lambda c: c.get("id", 0))
    cat_ids = [c["id"] for c in cats]
    class_names = [c["name"] for c in cats]
    return cat_ids, class_names


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean/0/1, got '{v}'")

def parse_patterns(s):
    if s is None or s == "":
        return None
    if isinstance(s, (list, tuple)):
        return list(s)
    # comma-separated string → list
    return [p.strip() for p in str(s).split(",") if p.strip()]

def pick_amp_for_gpu():
    """Use fp16 on pre-Ampere (like GTX 1650 Ti), bf16 on Ampere+."""
    if not torch.cuda.is_available():
        return True, "cpu"  # enable AMP anyway; engine guards it safely
    major, minor = torch.cuda.get_device_capability()
    # Pre-Ampere (<8.0) → float16; Ampere+ → bfloat16 (engine picks dtype)
    return True, "cuda"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["nano","small","base","medium","large"], default="small")
    ap.add_argument("--epochs", type=int, default=1, help="run 1 epoch locally to sanity-check")
    ap.add_argument("--batch",  type=int, default=1, help="micro-batch per step")
    ap.add_argument("--accum",  type=int, default=4, help="grad accumulation steps")
    ap.add_argument("--workers",type=int, default=2)
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    # booleans that accept 0/1 or true/false
    ap.add_argument("--multi_scale", type=str2bool, default=False, nargs="?", const=True,
                    help="use multi-scale training (accepts 0/1/true/false)")
    ap.add_argument("--expanded_scales", type=str2bool, default=False, nargs="?", const=True,
                    help="use expanded scale set (accepts 0/1/true/false)")
    # comma-separated glob patterns to exclude when loading pretrain
    ap.add_argument("--pretrain_exclude_keys", type=str, default="backbone.0.projector.*",
                    help='comma-separated patterns to skip, e.g. "backbone.0.projector.*,transformer.*"')
    args = ap.parse_args()

    size = args.size
    ModelCls = SIZE2CLS[size]
    defaults = SIZE_DEFAULTS[size]

    # Relative paths (work on Windows/Colab/Linux)
    dataset_dir = REPO_ROOT / "dataset"   # expects train/ valid/ test/ with _annotations.coco.json
    base_name = f"detector_{size}_{defaults['resolution']}"
    for i in count(1):
        candidate = REPO_ROOT / "runs" / f"{base_name}_{i}"
        if not candidate.exists():
            run_dir = candidate
            break

    amp_enabled, device = pick_amp_for_gpu()
    if args.no_amp:
        amp_enabled = False

    exclude_list = parse_patterns(args.pretrain_exclude_keys)

    # Validate dataset exists
    train_ann = dataset_dir / "train" / "_annotations.coco.json"
    if not train_ann.exists():
        raise FileNotFoundError(
            f"Training annotations not found: {train_ann}\n"
            f"Expected structure: {dataset_dir}/train/_annotations.coco.json"
        )
    
    _, class_names = read_coco_categories(str(train_ann))
    num_classes = len(class_names)
    exclude_list = (exclude_list or []) + ["class_embed.*"]  # drop pretrained cls head

    # Instantiate the chosen size with its own baked config.
    # IMPORTANT: Don’t override encoder/out_feature_indexes/projector_scale unless you know why.
    model = ModelCls(
        dataset_dir=str(dataset_dir),
        output_dir=str(run_dir),
        epochs=args.epochs,
        resolution=defaults["resolution"],
        pretrain_weights=defaults["pretrain_weights"],
        num_classes=num_classes,
        class_names=class_names,
        pretrain_exclude_keys=exclude_list,
        # Memory-friendly toggles for 4 GB:
        amp=amp_enabled,                 # engine will choose fp16 on pre-Ampere
        batch_size=args.batch,           # start tiny; scale with --accum
        grad_accum_steps=args.accum,     # raises effective batch without peaking VRAM
        num_workers=args.workers,
        use_ema=False,                   # EMA doubles model memory; disable for 4 GB
        aux_loss=False,                  # slightly reduces memory on decoder
        multi_scale=bool(args.multi_scale),
        expanded_scales=bool(args.expanded_scales),
        do_random_resize_via_padding=False,
        square_resize_div_64=(size in {"nano","small","medium"}),  # Base/Large use block size 56
        masks=False,                     # stage-1 detection only
        early_stopping=True,            # not needed for a sanity run
        lr_scheduler="cosine",
        warmup_epochs=0,
        device=device,
        run_test=False,                  # skip test pass to save memory/time
    )

    print("\n" + "="*80)
    print("STAGE 1: DETECTOR TRAINING")
    print("="*80)
    print(f"Model Size:       {size}")
    print(f"Resolution:       {defaults['resolution']}")
    print(f"Epochs:           {args.epochs}")
    print(f"Device:           {device}")
    print(f"Dataset:          {dataset_dir}")
    print(f"Output:           {run_dir}")
    print(f"Classes ({num_classes}):  {class_names}")
    print("="*80)
    print(f"Batch size:       {args.batch}")
    print(f"Grad accumulation: {args.accum}")
    print(f"Effective batch:  {args.batch * args.accum}")
    print(f"Workers:          {args.workers}")
    print(f"AMP:              {amp_enabled}")
    print(f"Multi-scale:      {bool(args.multi_scale)}")
    print(f"Expanded scales:  {bool(args.expanded_scales)}")
    print(f"Exclude keys:     {exclude_list}")
    print("="*80 + "\n")

    writer = SummaryWriter(str(run_dir / "tb"))

    # Kick off training (the model wrapper re-populates args internally)
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        num_workers=args.workers,
        output_dir=str(run_dir),
        amp=amp_enabled,
        aux_loss=False,
        use_ema=False,
        multi_scale=bool(args.multi_scale),
        expanded_scales=bool(args.expanded_scales),
        pretrain_exclude_keys=exclude_list,
        num_classes=num_classes,
        class_names=class_names,
        do_random_resize_via_padding=False,
        square_resize_div_64=(size in {"nano","small","medium"}),  # Consistent with model init
        lr_scheduler="cosine",
        warmup_epochs=0,
        run_test=False,
        masks=False,
        tensorboard_writer=writer
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Checkpoints saved to: {run_dir}")
    print(f"TensorBoard logs:     {run_dir / 'tb'}")
    print(f"\nTo view training: tensorboard --logdir {run_dir / 'tb'}")
    print("="*80 + "\n")
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
