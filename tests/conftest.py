"""Load host C++/zlib before test modules import jax."""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

from ipg._host import preload

preload()

import jax

jax.config.update("jax_enable_x64", True)
