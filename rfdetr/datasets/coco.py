# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import numpy as np
import torch
import torch.utils.data
import torchvision
from typing import Optional, Dict

from pycocotools import mask as coco_mask
import rfdetr.datasets.transforms as T

def compute_multi_scale_scales(resolution, expanded_scales=False, patch_size=16, num_windows=4):
    # round to the nearest multiple of 4*patch_size to enable both patching and windowing
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    proposed_scales = [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]  # ensure minimum image size
    return proposed_scales

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks: bool = False, remap_categories: bool = True):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        # Build contiguous cat mapping if requested
        self.cat2label: Optional[Dict[int, int]] = None
        if remap_categories:
            cat_ids = sorted(self.coco.getCatIds())
            self.cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        self.prepare = ConvertCoco(return_masks=return_masks, cat2label=self.cat2label)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCoco(object):
    def __init__(self, return_masks: bool = False, cat2label: Optional[Dict[int, int]] = None):
        self.return_masks = return_masks
        self.cat2label = cat2label

    def _ann_to_mask(self, ann, h, w):
        seg = ann.get("segmentation", None)
        if seg is None:
            return np.zeros((h, w), dtype=np.uint8)
        if isinstance(seg, list):  # polygons
            rles = coco_mask.frPyObjects(seg, h, w)
            rle = coco_mask.merge(rles)
        elif isinstance(seg, dict) and "counts" in seg:  # RLE
            rle = seg
        else:
            return np.zeros((h, w), dtype=np.uint8)
        m = coco_mask.decode(rle)  # (H,W) or (H,W,1)
        if m.ndim == 3:
            m = m[..., 0]
        return (m > 0).astype(np.uint8)

    def __call__(self, image, target):
        w, h = image.size
        image_id = int(target["image_id"])

        anno = target["annotations"]
        # drop crowds for instance training (you can keep them if desired)
        anno = [obj for obj in anno if obj.get('iscrowd', 0) == 0]

        # boxes: COCO bbox is [x,y,w,h] in px → make xyxy in px
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # to xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # labels: map to contiguous space if provided
        raw_labels = [obj["category_id"] for obj in anno]
        if self.cat2label is not None:
            classes = torch.tensor([self.cat2label[c] for c in raw_labels], dtype=torch.int64)
        else:
            classes = torch.tensor(raw_labels, dtype=torch.int64)

        # keep only valid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        out = {
            "boxes": boxes,              # xyxy in px (your transforms can convert/normalize later)
            "labels": classes,
            "image_id": torch.as_tensor(image_id, dtype=torch.int64),
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
        }

        # extra fields
        area = torch.tensor([obj.get("area", 0.0) for obj in anno])
        iscrowd = torch.tensor([obj.get("iscrowd", 0) for obj in anno])
        out["area"] = area[keep]
        out["iscrowd"] = iscrowd[keep]

        # masks (optional)
        if self.return_masks:
            if len(anno) == 0 or keep.sum().item() == 0:
                masks = torch.zeros((0, h, w), dtype=torch.uint8)
            else:
                kept_anno = [a for a, k in zip(anno, keep.tolist()) if k]
                masks_np = [self._ann_to_mask(a, h, w) for a in kept_anno]  # list of (H,W) uint8
                masks = torch.as_tensor(np.stack(masks_np, axis=0), dtype=torch.uint8)  # (N,H,W)
            out["masks"] = masks

        return image, out

def make_coco_transforms(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [resolution]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([resolution], max_size=1333),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([resolution], max_size=1333),
            normalize,
        ])
    if image_set == 'val_speed':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def make_coco_transforms_square_div_64(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):
    """
    """

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    scales = [resolution]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.SquareResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.SquareResize(scales),
                ]),
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])
    if image_set == 'test':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])
    if image_set == 'val_speed':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args, resolution):
    # Prefer dataset_dir (Roboflow-style). Fall back to coco_path only if provided.
    root_str = getattr(args, "dataset_dir", None) or getattr(args, "coco_path", None)
    assert root_str is not None, "Set args.dataset_dir to your dataset root containing train/valid/test"
    root = Path(root_str)
    assert root.exists(), f'provided dataset path {root} does not exist'

    # Support 'val_speed' by deriving the base split name, and map 'val' -> 'valid' if that folder exists
    base_split = image_set.split("_")[0]  # 'train' | 'val' | 'test'
    if base_split == "val" and (root / "valid").exists():
        split = "valid"
    else:
        split = base_split

    img_folder = root / split
    ann_file = img_folder / "_annotations.coco.json"
    assert img_folder.exists(), f"Images folder not found: {img_folder}"
    assert ann_file.exists(), f"COCO annotations not found: {ann_file}"

    tfm = make_coco_transforms_square_div_64 if getattr(args, "square_resize_div_64", False) else make_coco_transforms
    dataset = CocoDetection(
        img_folder, ann_file,
        transforms=tfm(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ),
        return_masks=getattr(args, "masks", False),
        remap_categories=True,
    )
    return dataset

def build_roboflow(image_set, args, resolution):
    root = Path(args.dataset_dir)
    assert root.exists(), f'provided Roboflow path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val":   (root / "valid", root / "valid" / "_annotations.coco.json"),
        "test":  (root / "test",  root / "test"  / "_annotations.coco.json"),
    }
    img_folder, ann_file = PATHS[image_set.split("_")[0]]

    tfm = make_coco_transforms_square_div_64 if getattr(args, "square_resize_div_64", False) else make_coco_transforms
    dataset = CocoDetection(
        img_folder, ann_file,
        transforms=tfm(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ),
        return_masks=getattr(args, "masks", False),
        remap_categories=True,
    )
    return dataset
