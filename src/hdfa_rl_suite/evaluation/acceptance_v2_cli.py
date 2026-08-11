from __future__ import annotations

import argparse
from pathlib import Path

from .acceptance_v2 import reconstruct_acceptance_v2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Losslessly reconstruct and audit the split acceptance-v2 report")
    parser.add_argument("--part-1", type=Path, default=Path(
        "artifacts/acceptance/compute-aware-v2/authoritative-comparison-v2.part-1.json.gz"))
    parser.add_argument("--part-2", type=Path, default=Path(
        "artifacts/acceptance/compute-aware-v2/authoritative-comparison-v2.part-2.json.gz"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/development"))
    args = parser.parse_args(argv)
    report = reconstruct_acceptance_v2(
        (args.part_1, args.part_2), args.output)
    print(args.output/"v2_reconstruction.json")
    print(args.output/"v2_reconstruction.md")
    print(f"passed={report['passed']} source_sha256={report['reconstructed_source_sha256']}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

