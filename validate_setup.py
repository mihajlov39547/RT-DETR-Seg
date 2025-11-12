"""
validate_setup.py - Validate dataset and environment before training

This script checks that your environment is ready for RF-DETR-Seg training.

Usage:
python validate_setup.py
"""
import sys
from pathlib import Path
import json

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def check_python_version():
    """Check Python version is compatible."""
    print("\n" + "="*80)
    print("CHECKING PYTHON VERSION")
    print("="*80)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required!")
        return False
    print("✅ Python version OK")
    return True


def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    print("\n" + "="*80)
    print("CHECKING PYTORCH")
    print("="*80)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  No CUDA GPU detected - training will be slow on CPU")
        print("✅ PyTorch OK")
        return True
    except ImportError:
        print("❌ PyTorch not installed!")
        print("Install: pip install torch torchvision")
        return False


def check_dependencies():
    """Check required packages are installed."""
    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    required = [
        "torch",
        "torchvision",
        "numpy",
        "PIL",
        "pycocotools",
        "supervision",
        "tensorboard",
    ]
    
    missing = []
    for package in required:
        try:
            if package == "PIL":
                import PIL
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies OK")
    return True


def check_dataset_structure():
    """Check dataset directory structure."""
    print("\n" + "="*80)
    print("CHECKING DATASET STRUCTURE")
    print("="*80)
    
    dataset_dir = REPO_ROOT / "dataset"
    print(f"Dataset directory: {dataset_dir}")
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("\nCreate the directory and add your COCO format data:")
        print("  dataset/")
        print("    ├── train/")
        print("    │   ├── _annotations.coco.json")
        print("    │   └── *.jpg")
        print("    ├── valid/")
        print("    │   ├── _annotations.coco.json")
        print("    │   └── *.jpg")
        return False
    
    print(f"✅ Dataset directory exists")
    
    # Check train split
    train_dir = dataset_dir / "train"
    train_ann = train_dir / "_annotations.coco.json"
    
    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return False
    print(f"✅ Train directory exists")
    
    if not train_ann.exists():
        print(f"❌ Train annotations not found: {train_ann}")
        return False
    print(f"✅ Train annotations exist")
    
    # Check valid split
    valid_dir = dataset_dir / "valid"
    valid_ann = valid_dir / "_annotations.coco.json"
    
    if not valid_dir.exists():
        print(f"⚠️  Valid directory not found: {valid_dir} (optional but recommended)")
    else:
        print(f"✅ Valid directory exists")
        if valid_ann.exists():
            print(f"✅ Valid annotations exist")
        else:
            print(f"⚠️  Valid annotations not found: {valid_ann}")
    
    return True


def check_annotations(check_masks=True):
    """Check COCO annotation file format."""
    print("\n" + "="*80)
    print("CHECKING ANNOTATIONS FORMAT")
    print("="*80)
    
    dataset_dir = REPO_ROOT / "dataset"
    train_ann = dataset_dir / "train" / "_annotations.coco.json"
    
    if not train_ann.exists():
        print("❌ Cannot check annotations - file not found")
        return False
    
    try:
        with open(train_ann, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
        return False
    
    # Check structure
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        if key not in data:
            print(f"❌ Missing required key: '{key}'")
            return False
        print(f"✅ Has '{key}' field")
    
    # Check categories
    categories = data.get("categories", [])
    num_classes = len(categories)
    class_names = [c["name"] for c in categories]
    print(f"\n📊 Dataset Info:")
    print(f"   Classes: {num_classes}")
    print(f"   Names: {class_names}")
    print(f"   Images: {len(data.get('images', []))}")
    print(f"   Annotations: {len(data.get('annotations', []))}")
    
    if num_classes == 0:
        print("❌ No categories found in annotations!")
        return False
    
    # Check annotations structure
    annotations = data.get("annotations", [])
    if not annotations:
        print("❌ No annotations found!")
        return False
    
    # Check for bounding boxes
    sample_ann = annotations[0]
    has_bbox = "bbox" in sample_ann
    print(f"\n✅ Bounding boxes: {'Present' if has_bbox else '❌ MISSING'}")
    
    # Check for segmentation masks (required for Stage 2)
    if check_masks:
        sample_size = min(10, len(annotations))
        mask_count = sum(1 for ann in annotations[:sample_size] 
                        if "segmentation" in ann and ann["segmentation"])
        
        if mask_count > 0:
            print(f"✅ Segmentation masks: Present ({mask_count}/{sample_size} samples checked)")
            print("   ✅ Ready for Stage 2 (segmentation training)")
        else:
            print(f"⚠️  Segmentation masks: NOT FOUND")
            print("   ⚠️  Stage 1 (detector) only - Stage 2 will fail without masks!")
            print("   ℹ️  To train segmentation, use instance annotations with polygons")
    
    return True


def check_pretrained_weights():
    """Check if pretrained weights are available."""
    print("\n" + "="*80)
    print("CHECKING PRETRAINED WEIGHTS")
    print("="*80)
    
    weights = {
        "rf-detr-nano.pth": "Nano model weights",
        "rf-detr-small.pth": "Small model weights",
        "rf-detr-base.pth": "Base model weights",
        "rf-detr-medium.pth": "Medium model weights",
        "rf-detr-large.pth": "Large model weights",
    }
    
    found = []
    missing = []
    
    for weight_file, desc in weights.items():
        weight_path = REPO_ROOT / weight_file
        if weight_path.exists():
            print(f"✅ {weight_file} - {desc}")
            found.append(weight_file)
        else:
            print(f"⚠️  {weight_file} - Not found (will auto-download on first use)")
            missing.append(weight_file)
    
    if found:
        print(f"\n✅ Found {len(found)} pretrained weight files")
    else:
        print("\nℹ️  No pretrained weights found - they will be downloaded automatically")
    
    return True


def main():
    """Run all validation checks."""
    print("\n" + "="*80)
    print("RF-DETR-SEG SETUP VALIDATION")
    print("="*80)
    print(f"Repository: {REPO_ROOT}")
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch", check_pytorch),
        ("Dependencies", check_dependencies),
        ("Dataset Structure", check_dataset_structure),
        ("Annotations", lambda: check_annotations(check_masks=True)),
        ("Pretrained Weights", check_pretrained_weights),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready for training!")
        print("="*80)
        print("\nQuick test commands:")
        print("\nStage 1 (Detector):")
        print("  python train_detector_stage1.py --size nano --epochs 1 --batch 1 --accum 4 --workers 0 --multi_scale 0")
        print("\nStage 2 (Segmentation):")
        print("  python train_seg_stage2.py --size nano --stage1_run runs/detector_nano_384_1 --epochs 1 --batch 1 --accum 2 --workers 0 --multi_scale 0 --run_test 0")
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before training")
        print("="*80)
        print("\nRefer to QUICK_START_GUIDE.md for setup instructions")
    
    print("\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
