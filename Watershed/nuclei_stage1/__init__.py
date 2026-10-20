"""Stage 1 (percentile mask -> distance transform -> seeded watershed) nuclei segmentation for tiled lightsheet volumes."""
from .config import Stage1Config
from .segmentation import run_stage1

__all__ = ["Stage1Config", "run_stage1"]
__version__ = "0.3.0"
