"""
train_seg_stage2.py

Stage-2 segmentation training for RF-DETR.

Overview:
---------
Stage-2 is the second training phase. In Stage-1 you trained the detector
(backbone + transformer + detection head) to predict boxes/classes.
In Stage-2 we reuse those detector weights, wrap the model with `DETRsegm`,
freeze the detector, and train only a new mask head to predict instance masks.

Main features:
--------------
- Loads a Stage-1 detector checkpoint (`--stage1_ckpt`).
- Wraps with `DETRsegm`, adds a convolutional mask head.
- Freezes detector weights (only mask head is trainable).
- Trains with COCO-style instance masks (polygon/RLE).
- Supports speed/VRAM control via `--num_queries`, `--mask_chunk`, etc.

Important notes:
----------------
* Your dataset must be in COCO format with segmentation masks.
* Class set and order must match Stage-1 (or training will fail).
* By default, the detector is frozen (fine-tune only the mask head).
* GTX 1650 Ti / 4 GB users should keep image res small (e.g. 256–384) and
  reduce queries/chunk size for faster runs.

Key arguments:
--------------
--size           Model size (nano/small/base/medium/large).
--stage1_ckpt    Path to Stage-1 checkpoint.
--output_dir     Where to save Stage-2 checkpoints/logs.
--resolution     Input image size (should match Stage-1).
--epochs         Number of epochs (default=50).
--batch          Micro-batch size.
--accum          Gradient accumulation steps.
--workers        Dataloader workers (set 0 on Windows).
--num_queries    Number of object queries (fewer = faster/less accurate).
--mask_chunk     Query chunk size in mask head (smaller = lower VRAM).
--multi_scale    Enable multi-scale augmentation (0/1).
--expanded_scales Enable expanded scale set (0/1).

Usage examples:
---------------
# Quick 1-epoch smoke run (faster, 200 iterations only):
python train_seg_stage2.py --size nano --stage1_ckpt runs/detector_nano_384_2/checkpoint_best_total_2class.pth \
    --output_dir smoke_seg --resolution 256 --epochs 1 --batch 1 --accum 1 --workers 0 \
    --num_queries 100 --mask_chunk 16 --multi_scale 0 --expanded_scales 0

# Full training run (frozen detector, 50 epochs at res=384):
python train_seg_stage2.py --size nano --stage1_ckpt runs/detector_nano_384_2/checkpoint_best_total_2class.pth \
    --output_dir output_seg_nano --resolution 384 --epochs 50 --batch 2 --accum 4 --workers 2
"""


import argparse
import json
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from rfdetr.models.segmentation import DETRsegm

# Safe matmul precision
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
cudnn.benchmark = True  # faster on fixed input sizes

# Core RF-DETR APIs
from rfdetr.detr import RFDETRBase, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRNano
from rfdetr.config import TrainConfig

SIZES = {
    "nano":   RFDETRNano,
    "small":  RFDETRSmall,
    "base":   RFDETRBase,
    "medium": RFDETRMedium,
    "large":  RFDETRLarge,
}

def read_coco_categories(ann_path: str):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data.get("categories", []), key=lambda c: c.get("id", 0))
    class_names = [c["name"] for c in cats]
    return class_names, len(class_names)

def parse_args():
    p = argparse.ArgumentParser("RF-DETR Stage-2 (Segmentation Head) Trainer")

    # Required
    p.add_argument("--stage1_ckpt", type=str, required=True,
                   help="Stage-1 detector checkpoint (e.g. runs/.../checkpoint_best_total.pth)")

    # Common
    p.add_argument("--size", type=str, default="base", choices=list(SIZES.keys()))
    p.add_argument("--output_dir", type=str, default="output_seg")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resolution", type=int, default=560,
                   help="Should match Stage-1 unless you know what you’re doing")

    # Train hyperparams
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5, help="LR for mask head (detector frozen)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)

    # Loss weights
    p.add_argument("--mask_loss_coef", type=float, default=2.0)
    p.add_argument("--dice_loss_coef", type=float, default=2.0)
    p.add_argument("--freeze", type=int, default=1,
               help="1 = freeze detector (mask head only), 0 = train detector + mask head")
    p.add_argument("--cls_loss_coef", type=float, default=1.0)
    p.add_argument("--bbox_loss_coef", type=float, default=5.0)
    p.add_argument("--giou_loss_coef", type=float, default=2.0)

    # Extras
    p.add_argument("--ema", action="store_true", help="Use EMA for the (frozen) wrapper")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--eval_only", action="store_true", help="Just run validation once")
    p.add_argument("--run_test", action="store_true", help="Evaluate on test split after training")

    # Speed/compute knobs
    p.add_argument("--num_queries", type=int, default=300, help="Number of object queries (default=300)")
    p.add_argument("--mask_chunk", type=int, default=32, help="Chunk size for mask head forward (default=32)")
    p.add_argument("--multi_scale", type=int, default=0, help="Enable multi-scale aug (0/1)")
    p.add_argument("--expanded_scales", type=int, default=0, help="Enable expanded scales (0/1)")

    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_dir = Path(r"/content/drive/MyDrive/RT-DETR-Seg/dataset").resolve()
    train_ann = dataset_dir / "train" / "_annotations.coco.json"
    if not train_ann.exists():
        raise FileNotFoundError(f"Missing annotations: {train_ann}")

    # Require masks present
    with open(train_ann, "r", encoding="utf-8") as f:
        anns = json.load(f)
    if not any(a.get("segmentation") for a in anns.get("annotations", [])):
        raise ValueError("This Stage-2 script requires instance masks.")

    # Read classes
    class_names, num_classes = read_coco_categories(str(train_ann))

    # Stage-1 checkpoint sanity check (class count)
    try:
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model") or ckpt.get("ema_model") or {}
        ckpt_args = ckpt.get("args", None)
        cls_bias_key = next((k for k in state.keys() if k.endswith("class_embed.bias")), None)
        if cls_bias_key is not None:
            ckpt_out = int(state[cls_bias_key].shape[0])   # includes background
            ds_out   = int(num_classes + 1)
            if ckpt_out != ds_out:
                raise ValueError(
                    f"Stage-1 checkpoint classes ({ckpt_out-1}) != dataset classes ({num_classes})."
                )
    except Exception:
        pass

    # Build base RF-DETR
    ModelCls = SIZES[args.size]
    # decide freezing: only freeze if --freeze=1
    _frozen = args.stage1_ckpt if args.freeze == 1 else None
    # prefer Stage-1 architecture settings when present
    def _get(k, default):
        return getattr(ckpt_args, k, default) if ckpt_args is not None else default
    model = ModelCls(
        resolution=_get("resolution", args.resolution),
        masks=True,
        frozen_weights=_frozen,
        device=args.device,
        num_classes=num_classes,
        class_names=class_names,
        aux_loss=True,  # enable decoder aux supervision for seg
        num_queries=_get("num_queries", args.num_queries),
        # out_feature_indexes=_get("out_feature_indexes", None),
        # projector_scale=_get("projector_scale", None),
        # position_embedding=_get("position_embedding", None),
        multi_scale=bool(args.multi_scale),
        expanded_scales=bool(args.expanded_scales),
    )

    freeze_intent = (args.freeze == 1)
    inner = model.model.model
    if not isinstance(inner, DETRsegm):
        print("[stage-2] Wrapping base detector with DETRsegm.")
        model.model.model = DETRsegm(inner, freeze_detr=freeze_intent,
                                     mask_chunk=args.mask_chunk).to(model.model.device)

    # Train config
    tcfg = TrainConfig(
        dataset_dir=str(dataset_dir),
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        num_workers=args.workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tensorboard=args.tensorboard,
        use_ema=args.ema,
        run_test=args.run_test and (not args.eval_only),
        aux_loss=True,
        # losses
        cls_loss_coef=args.cls_loss_coef,
        bbox_loss_coef=args.bbox_loss_coef,
        giou_loss_coef=args.giou_loss_coef,
        mask_loss_coef=args.mask_loss_coef,
        dice_loss_coef=args.dice_loss_coef,
        masks=True,
        frozen_weights=_frozen,
        class_names=class_names,
        square_resize_div_64=True,
        # keep lightweight by default
        multi_scale=bool(args.multi_scale),
        expanded_scales=bool(args.expanded_scales),
        do_random_resize_via_padding=False,
        # housekeeping
        seed=args.seed,
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_use_ema=False,
    )

    if args.eval_only:
        model.model.args.eval = True
        model.train_from_config(tcfg, eval=True, run_test=False)
        return

    print(f"==> Stage-2 (seg) training on {args.device} | size={args.size} | res={args.resolution}")
    print(f"    dataset_dir={dataset_dir}")
    print(f"    stage1_ckpt={args.stage1_ckpt}")
    print(f"    classes({num_classes})={class_names}")
    
    model.train_from_config(tcfg)

    print("\nStage-2 training complete.")
    print(f"Artifacts in: {args.output_dir}")

if __name__ == "__main__":
    main()
