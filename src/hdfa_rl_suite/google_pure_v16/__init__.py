"""V16 source-normalized optimizer-consistency diagnostics.

Every result is reduced, development-only evidence.  Importing this package never
launches acquisition and never opens a held-out or source-budget seed registry.
"""

from .contracts import NONFINAL, V16_SCHEMA

__all__ = ["NONFINAL", "V16_SCHEMA"]
