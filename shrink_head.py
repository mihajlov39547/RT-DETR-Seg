"""
shrink_head.py

Usage:
python shrink_head.py --in_ckpt runs/detector_nano_384_2/checkpoint_best_total.pth --out_ckpt runs/detector_nano_384_2/checkpoint_best_total_2class.pth --keep "blood_vessel" "glomerulus"
"""

import argparse, torch, json, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--keep", nargs="+", required=True,
                   help="Class names to keep (foreground only). Background is kept automatically.")
    args = ap.parse_args()

    ckpt = torch.load(args.in_ckpt, map_location="cpu")
    state = ckpt.get("model", {})
    if not state:
        sys.exit("No 'model' in checkpoint")

    # Try to get class names
    ckpt_classes = None
    if "args" in ckpt and hasattr(ckpt["args"], "class_names"):
        ckpt_classes = list(ckpt["args"].class_names)

    # Find class head bias
    cls_bias_key = next((k for k in state.keys() if k.endswith("class_embed.bias")), None)
    if cls_bias_key is None:
        sys.exit("Couldn't find class head bias in checkpoint")
    old_out = state[cls_bias_key].shape[0]
    bg_idx = old_out - 1  # DETR: background last

    # Map keep names -> indices
    if ckpt_classes is None:
        print("Warning: checkpoint has no args.class_names; "
              "assuming your --keep order matches old head rows (COCO order).")
        keep_rows = list(range(len(args.keep)))
    else:
        name_to_idx = {n: i for i, n in enumerate(ckpt_classes)}
        missing = [n for n in args.keep if n not in name_to_idx]
        if missing:
            sys.exit(f"These classes were not found in checkpoint class_names: {missing}")
        keep_rows = [name_to_idx[n] for n in args.keep]

    new_order = keep_rows + [bg_idx]
    new_out = len(new_order)

    def slice_head_rows(w, rows):
        return w.index_select(0, torch.tensor(rows, dtype=torch.long))

    num_sliced = 0
    for k in list(state.keys()):
        if k.endswith("class_embed.weight") or k.endswith("class_embed.bias"):
            state[k] = slice_head_rows(state[k], new_order)
            num_sliced += 1

    if num_sliced == 0:
        sys.exit("No class_embed tensors found to slice; aborting.")

    # Update args
    if "args" in ckpt:
        try:
            ckpt["args"].num_classes = new_out - 1
        except Exception:
            pass
        try:
            ckpt["args"].class_names = args.keep
        except Exception:
            pass

    torch.save(ckpt, args.out_ckpt)
    print(f"Done. Old out_dim={old_out} -> new out_dim={new_out}. "
          f"Sliced {num_sliced} tensors.\nWrote: {args.out_ckpt}")

    # Manual verification
    ckpt = torch.load(args.out_ckpt, map_location="cpu")
    w = ckpt["model"]
    key = [k for k in w if k.endswith("class_embed.bias")][0]
    print("Verification: out_dim =", w[key].shape[0],
          "| expected =", len(args.keep) + 1)


if __name__ == "__main__":
    main()
