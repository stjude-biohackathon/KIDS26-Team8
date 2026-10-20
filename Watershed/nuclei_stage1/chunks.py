"""
Chunk discovery for a slab folder laid out as

    <slab>/raw_chunked/{roi_stem}_z{Z}_y{Y}_x{X}.tif     one 3-D tile per file

(Z, Y, X) in a tile's name is its offset INSIDE its source ROI file, not a slab-global coordinate: every ROI
restarts at z0_y0_x0, and a slab usually holds several ROIs. Tiles are therefore keyed by (roi_stem, z, y, x)
together; keying on the offset alone would silently collide tiles from different ROIs.
"""
import re
from dataclasses import dataclass
from pathlib import Path

CHUNK_RE = re.compile(r"^(?P<stem>.+)_z(?P<z>\d+)_y(?P<y>\d+)_x(?P<x>\d+)$")


@dataclass
class ChunkPaths:
    name: str
    roi_stem: str
    z: int
    y: int
    x: int
    raw_tif: Path


def parse_chunk_name(stem):
    """Split a tile filename stem into (roi_stem, z, y, x); None if it does not match the naming."""
    m = CHUNK_RE.match(stem)
    if not m:
        return None
    return m["stem"], int(m["z"]), int(m["y"]), int(m["x"])


def list_chunks(slab_dir):
    """Every tile in <slab_dir>/raw_chunked/, sorted by (roi_stem, z, y, x)."""
    raw_dir = Path(slab_dir) / "raw_chunked"
    out = []
    if not raw_dir.is_dir():
        return out
    for p in sorted(raw_dir.glob("*.tif")):
        parsed = parse_chunk_name(p.stem)
        if parsed:
            roi, z, y, x = parsed
            out.append(ChunkPaths(name=p.stem, roi_stem=roi, z=z, y=y, x=x, raw_tif=p))
    return sorted(out, key=lambda c: (c.roi_stem, c.z, c.y, c.x))
