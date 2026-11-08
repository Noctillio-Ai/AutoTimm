"""Shared test configuration."""

import os

import matplotlib

# Keep `import autotimm` from querying PyPI during tests.
os.environ.setdefault("AUTOTIMM_DISABLE_UPDATE_CHECK", "1")

# Use non-interactive backend so tests work in headless CI (no display/Tk).
matplotlib.use("Agg")
