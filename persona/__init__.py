"""Persona Explorer's flat source tree, shared with the flat remote deployment.

Import this package before using legacy ``persona_*`` module names. Keeping one
module namespace preserves monkeypatches and the deployed helpers' source bytes.
CLI scripts run directly from this directory and need no bootstrap.
"""
from pathlib import Path
import sys

_SOURCE = str(Path(__file__).resolve().parent)
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)
