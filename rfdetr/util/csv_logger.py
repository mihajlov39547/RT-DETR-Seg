"""
util/csv_logger.py
"""
import csv, os, time
from typing import Dict, Iterable, Optional

def _to_float(v):
    try:
        import torch
        if isinstance(v, torch.Tensor):
            v = v.detach()
            return v.item() if v.numel() == 1 else float(v.mean())
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return v

def _flatten(prefix: str, d: Dict):
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flatten(key, v)
        else:
            yield key, _to_float(v)

class TidyCSVLogger:
    """
    - Auto-creates header from first row (no manual column list).
    - Safe to call repeatedly; writes header only if file empty.
    - Accepts nested dicts and tensors; flattens keys.
    - Separate train/val files recommended.
    """
    def __init__(self, path: str, extra_fieldnames: Optional[Iterable[str]] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = list(extra_fieldnames) if extra_fieldnames else None
        self._writer = None
        self._f = open(self.path, "a", newline="")
        self._header_written = (self._f.tell() != 0)

    def log(self, row: Dict):
        # Flatten tensors/nested dicts
        flat = dict(_flatten("", row))
        flat.setdefault("time", time.time())
        # Lazily initialize writer with union of keys
        if self._writer is None:
            if self.fieldnames is None:
                self.fieldnames = list(flat.keys())
            self._writer = csv.DictWriter(self._f, fieldnames=self.fieldnames, extrasaction="ignore")
            if not self._header_written:
                self._writer.writeheader()
                self._header_written = True
        # Grow header if new keys appear (rare but handy)
        new_keys = [k for k in flat.keys() if k not in self.fieldnames]
        if new_keys:
            self.fieldnames.extend(new_keys)
            # rewrite header only if file was empty; otherwise ignore—CSV readers handle missing cols
        self._writer.writerow(flat)
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
