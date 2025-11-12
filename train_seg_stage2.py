"""
train_seg_stage2.py - Stage 2: Train Segmentation Head with Frozen Detector

This script loads a trained detector checkpoint from Stage 1 and trains only
the segmentation head while keeping the detector frozen.

Usage:
python train_seg_stage2.py --size nano --stage1_run runs/detector_nano_384_1 --epochs 50 --batch 2 --accum 2 --workers 2
python train_seg_stage2.py --size base --stage1_run runs/detector_base_560_1 --epochs 50 --batch 1 --accum 4 --workers 0
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

# Handy defaults per size
SIZE_DEFAULTS = {
    "nano":   {"resolution": 384},
    "small":  {"resolution": 512},
    "base":   {"resolution": 560},
    "medium": {"resolution": 576},
    "large":  {"resolution": 560},
}

import json

def read_coco_categories(ann_path):
    """Read category information from COCO annotation file."""
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data.get("categories", []), key=lambda c: c.get("id", 0))
    cat_ids = [c["id"] for c in cats]
    class_names = [c["name"] for c in cats]
    return cat_ids, class_names


def validate_coco_has_masks(ann_path):
    """Verify that the COCO annotation file contains segmentation masks."""
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    annotations = data.get("annotations", [])
    if not annotations:
        raise ValueError(f"No annotations found in {ann_path}")
    
    # Check first 10 annotations for segmentation field
    sample_size = min(10, len(annotations))
    has_masks = sum(1 for ann in annotations[:sample_size] if "segmentation" in ann and ann["segmentation"])
    
    if has_masks == 0:
        raise ValueError(
            f"No segmentation masks found in {ann_path}!\n"
            "Stage 2 requires instance segmentation annotations.\n"
            "Your COCO JSON must have 'segmentation' field in annotations."
        )
    
    print(f"✓ Validated: {has_masks}/{sample_size} sample annotations contain segmentation masks")


def find_stage1_checkpoint(stage1_run_dir):
    """Find the best checkpoint from stage 1 training."""
    run_path = Path(stage1_run_dir)
    
    # Try common checkpoint names (both .pt and .pth extensions)
    candidates = [
        run_path / "best.pt",
        run_path / "best.pth",
        run_path / "checkpoint_best_regular.pth",
        run_path / "checkpoint_best_total.pth",
        run_path / "last.pt",
        run_path / "last.pth",
        run_path / "checkpoint.pt",
        run_path / "checkpoint.pth",
    ]
    
    for ckpt in candidates:
        if ckpt.exists():
            print(f"✓ Found Stage-1 checkpoint: {ckpt}")
            return ckpt
    
    # List all .pt and .pth files if none of the common names found
    pt_files = list(run_path.glob("*.pt")) + list(run_path.glob("*.pth"))
    if pt_files:
        ckpt = pt_files[0]  # Take first checkpoint file
        print(f"⚠ Using checkpoint: {ckpt}")
        return ckpt
    
    raise FileNotFoundError(
        f"No checkpoint found in {stage1_run_dir}!\n"
        f"Expected one of: {[c.name for c in candidates[:8]]}\n"
        f"Please verify Stage-1 training completed successfully."
    )


def verify_frozen_detector(model):
    """Verify that detector parameters are frozen and only seg head is trainable."""
    detector_params = []
    seg_head_params = []
    
    # Access the actual PyTorch model
    torch_model = model.model.model if hasattr(model.model, 'model') else model.model
    
    for name, param in torch_model.named_parameters():
        if "mask_head" in name or "bbox_attention" in name:
            seg_head_params.append((name, param.requires_grad))
        else:
            detector_params.append((name, param.requires_grad))
    
    # Check detector is frozen
    frozen_count = sum(1 for _, req_grad in detector_params if not req_grad)
    trainable_count = sum(1 for _, req_grad in detector_params if req_grad)
    
    if trainable_count > 0:
        print(f"⚠ WARNING: {trainable_count} detector parameters are trainable!")
        print("  Detector should be frozen in Stage-2. Check frozen_weights parameter.")
    else:
        print(f"✓ Detector frozen: {frozen_count} parameters")
    
    # Check seg head is trainable
    seg_trainable = sum(1 for _, req_grad in seg_head_params if req_grad)
    seg_frozen = sum(1 for _, req_grad in seg_head_params if not req_grad)
    
    if seg_trainable == 0:
        raise RuntimeError(
            "ERROR: Segmentation head has no trainable parameters!\n"
            "Check that masks=True and freeze_detr is properly set."
        )
    
    print(f"✓ Segmentation head trainable: {seg_trainable} parameters (frozen: {seg_frozen})")
    
    return {
        "detector_frozen": frozen_count,
        "detector_trainable": trainable_count,
        "seg_head_trainable": seg_trainable,
        "seg_head_frozen": seg_frozen,
    }


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean/0/1, got '{v}'")


def pick_amp_for_gpu():
    """Use fp16 on pre-Ampere (like GTX 1650 Ti), bf16 on Ampere+."""
    if not torch.cuda.is_available():
        return True, "cpu"  # enable AMP anyway; engine guards it safely
    major, minor = torch.cuda.get_device_capability()
    # Pre-Ampere (<8.0) → float16; Ampere+ → bfloat16 (engine picks dtype)
    return True, "cuda"


def main():
    ap = argparse.ArgumentParser(description="Stage 2: Train segmentation head with frozen detector")
    
    # Required
    ap.add_argument("--size", choices=["nano","small","base","medium","large"], required=True,
                    help="Model size (must match Stage-1)")
    ap.add_argument("--stage1_run", type=str, required=True,
                    help="Path to Stage-1 run directory (e.g., runs/detector_nano_384_1)")
    
    # Training params
    ap.add_argument("--epochs", type=int, default=50,
                    help="Number of epochs to train segmentation head")
    ap.add_argument("--batch",  type=int, default=2,
                    help="Batch size per step (segmentation needs more memory)")
    ap.add_argument("--accum",  type=int, default=2,
                    help="Gradient accumulation steps")
    ap.add_argument("--workers",type=int, default=2,
                    help="Number of dataloader workers")
    ap.add_argument("--no_amp", action="store_true",
                    help="Disable mixed precision training")
    
    # Segmentation-specific
    ap.add_argument("--seg_lr", type=float, default=1e-4,
                    help="Learning rate for segmentation head (lower than detector)")
    ap.add_argument("--mask_loss_coef", type=float, default=1.0,
                    help="Weight for mask loss")
    ap.add_argument("--dice_loss_coef", type=float, default=1.0,
                    help="Weight for dice loss")
    
    # Data augmentation
    ap.add_argument("--multi_scale", type=str2bool, default=True, nargs="?", const=True,
                    help="Use multi-scale training")
    ap.add_argument("--expanded_scales", type=str2bool, default=False, nargs="?", const=True,
                    help="Use expanded scale set")
    
    # Optional overrides
    ap.add_argument("--resolution", type=int, default=None,
                    help="Override resolution (default: use Stage-1 resolution)")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Specific checkpoint file to load (default: auto-find best.pt)")
    ap.add_argument("--run_test", type=str2bool, default=True, nargs="?", const=True,
                    help="Run test evaluation after training")
    
    args = ap.parse_args()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    size = args.size
    ModelCls = SIZE2CLS[size]
    defaults = SIZE_DEFAULTS[size]
    
    # Determine resolution
    resolution = args.resolution if args.resolution else defaults["resolution"]
    
    # Find Stage-1 checkpoint
    stage1_run_dir = Path(args.stage1_run)
    if not stage1_run_dir.exists():
        raise FileNotFoundError(f"Stage-1 run directory not found: {stage1_run_dir}")
    
    if args.checkpoint:
        stage1_checkpoint = Path(args.checkpoint)
        if not stage1_checkpoint.exists():
            raise FileNotFoundError(f"Specified checkpoint not found: {stage1_checkpoint}")
    else:
        stage1_checkpoint = find_stage1_checkpoint(stage1_run_dir)
    
    # Setup paths
    dataset_dir = REPO_ROOT / "dataset"
    base_name = f"segmentation_{size}_{resolution}"
    for i in count(1):
        candidate = REPO_ROOT / "runs" / f"{base_name}_{i}"
        if not candidate.exists():
            run_dir = candidate
            break
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # AMP setup
    amp_enabled, device = pick_amp_for_gpu()
    if args.no_amp:
        amp_enabled = False
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    print("\n" + "="*80)
    print("STAGE 2: SEGMENTATION HEAD TRAINING")
    print("="*80)
    print(f"Model Size:       {size}")
    print(f"Resolution:       {resolution}")
    print(f"Stage-1 Checkpoint: {stage1_checkpoint}")
    print(f"Output Directory: {run_dir}")
    print(f"Device:           {device}")
    print(f"AMP:              {amp_enabled}")
    print("="*80 + "\n")
    
    # Validate dataset has mask annotations
    train_ann = dataset_dir / "train" / "_annotations.coco.json"
    if not train_ann.exists():
        raise FileNotFoundError(f"Training annotations not found: {train_ann}")
    
    validate_coco_has_masks(train_ann)
    
    # Read class information
    _, class_names = read_coco_categories(str(train_ann))
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}\n")
    
    # =========================================================================
    # MODEL SETUP
    # =========================================================================
    
    print("Initializing model with frozen detector...")
    
    model = ModelCls(
        dataset_dir=str(dataset_dir),
        output_dir=str(run_dir),
        epochs=args.epochs,
        resolution=resolution,
        model_size=size,  # Pass model size for mask head selection
        
        # CRITICAL: Load Stage-1 checkpoint and freeze detector
        frozen_weights=str(stage1_checkpoint),
        masks=True,  # Enable segmentation head
        
        # Class information
        num_classes=num_classes,
        class_names=class_names,
        
        # Training hyperparameters (adjusted for seg head training)
        lr=args.seg_lr,  # Lower LR for fine-tuning seg head
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        num_workers=args.workers,
        
        # Memory management
        amp=amp_enabled,
        use_ema=False,  # Disable EMA to save memory
        aux_loss=True,  # Keep auxiliary losses for better training
        
        # Data augmentation
        multi_scale=bool(args.multi_scale),
        expanded_scales=bool(args.expanded_scales),
        do_random_resize_via_padding=False,
        square_resize_div_64=(size in {"nano","small","medium"}),
        
        # Loss weights
        mask_loss_coef=args.mask_loss_coef,
        dice_loss_coef=args.dice_loss_coef,
        
        # Training settings
        lr_scheduler="cosine",
        warmup_epochs=5,  # Short warmup for seg head
        early_stopping=True,
        patience=10,  # Stop if no improvement for 10 epochs
        
        device=device,
        run_test=bool(args.run_test),
    )
    
    # =========================================================================
    # VERIFICATION
    # =========================================================================
    
    print("\nVerifying model configuration...")
    param_stats = verify_frozen_detector(model)
    
    # Save configuration
    config_path = run_dir / "stage2_config.json"
    import json
    with open(config_path, "w") as f:
        json.dump({
            "stage": 2,
            "size": size,
            "resolution": resolution,
            "stage1_checkpoint": str(stage1_checkpoint),
            "epochs": args.epochs,
            "batch_size": args.batch,
            "grad_accum": args.accum,
            "learning_rate": args.seg_lr,
            "amp_enabled": amp_enabled,
            "device": device,
            "num_classes": num_classes,
            "class_names": class_names,
            "param_stats": param_stats,
            "args": vars(args),
        }, f, indent=2)
    print(f"✓ Configuration saved to: {config_path}\n")
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch}")
    print(f"Grad accumulation: {args.accum}")
    print(f"Effective batch:  {args.batch * args.accum}")
    print(f"Learning rate:    {args.seg_lr}")
    print(f"Workers:          {args.workers}")
    print(f"Multi-scale:      {bool(args.multi_scale)}")
    print("="*80 + "\n")
    
    writer = SummaryWriter(str(run_dir / "tb"))
    
    try:
        # Kick off training
        model.train(
            dataset_dir=str(dataset_dir),
            epochs=args.epochs,
            batch_size=args.batch,
            grad_accum_steps=args.accum,
            num_workers=args.workers,
            output_dir=str(run_dir),
            
            # Model config
            frozen_weights=str(stage1_checkpoint),
            masks=True,
            num_classes=num_classes,
            class_names=class_names,
            
            # Training settings
            lr=args.seg_lr,
            amp=amp_enabled,
            aux_loss=True,
            use_ema=False,
            multi_scale=bool(args.multi_scale),
            expanded_scales=bool(args.expanded_scales),
            
            # Loss weights
            mask_loss_coef=args.mask_loss_coef,
            dice_loss_coef=args.dice_loss_coef,
            
            # Other settings
            do_random_resize_via_padding=False,
            square_resize_div_64=(size in {"nano","small","medium"}),
            lr_scheduler="cosine",
            warmup_epochs=5,
            run_test=bool(args.run_test),
            
            tensorboard_writer=writer
        )
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Checkpoints saved to: {run_dir}")
        print(f"TensorBoard logs: {run_dir / 'tb'}")
        print(f"\nTo visualize training: tensorboard --logdir {run_dir / 'tb'}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR DURING TRAINING")
        print("="*80)
        print(f"{type(e).__name__}: {e}")
        print("="*80 + "\n")
        raise
    
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()