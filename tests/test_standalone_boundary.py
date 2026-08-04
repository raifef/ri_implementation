from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
PACKAGE = "google_rl_reimplementation"


def _token(codepoints: tuple[int, ...]) -> str:
    return "".join(chr(value) for value in codepoints)


def test_source_tree_uses_only_the_standalone_namespace():
    allowed = {PACKAGE}
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported = {node.module.split(".")[0]}
            else:
                continue
            local = {name for name in imported if name.startswith("google_")}
            assert local <= allowed, (path, local - allowed)


def test_project_has_no_prohibited_legacy_terminology():
    short = _token((104, 100, 102, 97))
    long = _token((104, 105, 101, 114, 97, 114, 99, 104, 105, 99, 97, 108, 32, 100, 105, 115, 99, 114, 101, 116, 101, 32, 102, 108, 117, 99, 116, 117, 97, 116, 105, 111, 110, 32, 97, 117, 116, 111, 115, 101, 103, 109, 101, 110, 116, 97, 116, 105, 111, 110))
    for path in ROOT.rglob("*"):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        relative = str(path.relative_to(ROOT)).lower()
        assert short not in relative and long not in relative
        if path.is_file() and "__pycache__" not in path.parts:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            assert short not in text and long not in text, path
