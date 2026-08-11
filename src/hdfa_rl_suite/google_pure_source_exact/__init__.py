"""Clean-room implementation of publicly identifiable Google RL mechanisms.

This namespace is deliberately isolated from the historical v5--v8 studies.  A
result is paper-comparable only when its emitted contract says so explicitly.
"""

from .control_normalization.contracts import SourceIdentifiability

__all__ = ["SourceIdentifiability"]

