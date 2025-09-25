"""Utility helpers for locating Lightning checkpoints."""
import glob
import os
from typing import Optional


def find_latest_checkpoint(directory: str) -> Optional[str]:
    """Return the most recent .ckpt file in the directory if it exists."""
    pattern = os.path.join(directory, "*.ckpt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)
