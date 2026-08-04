from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _standalone_smoke_prerequisites():
    from google_rl_reimplementation.bootstrap import run_compact_bootstrap

    run_compact_bootstrap()
