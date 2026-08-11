"""Module entry point for the exact-config benchmark preflight."""

from .cli import preflight_main


if __name__ == "__main__":
    raise SystemExit(preflight_main())
