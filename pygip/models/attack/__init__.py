"""
Attack module exports.

Some attacks depend on DGL. DGL wheels may be unavailable on some platforms (e.g., macOS).
We import DGL-dependent attacks conditionally so PyG-only workflows still work.
"""

from .egsteal import EGStealAttack

# Optional: if you KNOW any of these are PyG-only, you can import them here.
# For now, we keep everything else behind the DGL gate to avoid import-time crashes.


try:
    import dgl  # noqa: F401

    # Import ALL attacks that require DGL here:
    from .AdvMEA import AdvMEA
    from .CEGA import CEGA
    from .DataFreeMEA import DataFreeMEA
    from .Realistic import Realistic

except ImportError:
    AdvMEA = None
    CEGA = None
    DataFreeMEA = None
    Realistic = None

__all__ = [
    "EGStealAttack",
    "AdvMEA",
    "CEGA",
    "DataFreeMEA",
    "Realistic",
]
