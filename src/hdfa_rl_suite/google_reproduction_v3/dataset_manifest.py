"""Discovery, identity verification, and immutable ZIP-member inventory."""

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .schemas import OFFICIAL_RELEASE


def load_json_yaml(path: Path) -> dict[str, Any]:
    """Load configuration written as the JSON subset of YAML 1.2."""

    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _archive_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "resolved_path": str(path.resolve()).replace("\\", "/"),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def hash_archive_once(path: Path, cache_path: Path) -> dict[str, Any]:
    fingerprint = _archive_fingerprint(path)
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if all(cache.get(key) == value for key, value in fingerprint.items()):
            return {**cache, "cache_hit": True}

    md5 = hashlib.md5()  # noqa: S324 - required to verify the published Zenodo checksum.
    sha256 = hashlib.sha256()
    started = time.perf_counter()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    cache = {
        **fingerprint,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "hash_runtime_seconds": time.perf_counter() - started,
        "algorithms": ["md5", "sha256"],
    }
    _write_json(cache_path, cache)
    return {**cache, "cache_hit": False}


def discover_release(config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    selected = Path(config["selected_archive"])
    paths: set[Path] = {selected}
    for root_value in config.get("search_roots", []):
        root = Path(root_value)
        if root.is_dir():
            paths.update(root.glob("*google*reinforcement*learning*qec*.zip"))
    for path in sorted(paths, key=lambda item: str(item).lower()):
        if path.is_file():
            candidates.append(
                {
                    "path": str(path.resolve()).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "selected": path.resolve() == selected.resolve(),
                }
            )
    return candidates


def _member_role(path: str) -> tuple[str, str | None]:
    name = path.rstrip("/").split("/")[-1]
    lower = path.lower()
    if path.endswith("/"):
        return "directory", None
    if name == "README.md":
        return "release_readme", None
    if name == "logicals.png":
        return "documentation_figure", None
    if name == "metadata.json":
        return "experiment_metadata", None
    if name == "circuit_ideal.stim":
        return "ideal_annotated_circuit", None
    if name == "circuit_noisy_si1000.stim":
        return "si1000_noisy_circuit", None
    if name == "measurements.b8":
        return "raw_device_measurements", None
    if name == "sweep_bits.b8":
        return "circuit_configuration_bits", None
    if name == "detection_events.b8":
        return "derived_detection_events", None
    if name == "obs_flips_actual.b8":
        return "actual_logical_observable_flips", None
    if name == "obs_flips_predicted.b8":
        ambiguity = None
        if "/aq2-ens9/" in lower:
            ambiguity = "undocumented decoder-path alias; three members only"
        return "decoder_predicted_logical_flips", ambiguity
    if name == "error_model.dem":
        return "decoder_detector_error_model", None
    return "unclassified", "member name is not described by the release README"


def _file_type(path: str) -> str:
    if path.endswith("/"):
        return "directory"
    suffix = Path(path).suffix.lower()
    return {
        ".b8": "stim_b8",
        ".stim": "stim_circuit",
        ".dem": "stim_detector_error_model",
        ".json": "json",
        ".md": "markdown",
        ".png": "png",
    }.get(suffix, suffix.removeprefix(".") or "unknown")


def _parse_status(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[str, str | None]:
    if info.is_dir():
        return "DIRECTORY", None
    suffix = Path(info.filename).suffix.lower()
    if suffix == ".json":
        try:
            value = json.loads(archive.read(info))
            if info.filename.endswith("/metadata.json"):
                required = {"basis", "rounds", "shots", "qubit_coords"}
                missing = sorted(required - set(value))
                if missing:
                    return "MALFORMED", f"missing required keys: {missing}"
            return "PARSED", None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return "MALFORMED", str(exc)
    if suffix in {".md", ".stim", ".dem"}:
        try:
            with archive.open(info) as source:
                source.read(min(info.file_size, 64 * 1024)).decode("utf-8")
            return "UTF8_STREAMABLE", None
        except UnicodeDecodeError as exc:
            return "MALFORMED", str(exc)
    if suffix == ".b8":
        return "B8_STREAMABLE_REQUIRES_COMPANION_SCHEMA", None
    if suffix == ".png":
        return "BINARY_RECOGNIZED_NOT_PARSED", None
    return "UNKNOWN_TYPE_NOT_PARSED", "no registered parser"


def build_inventory(config_path: Path, artifact_dir: Path) -> dict[str, Any]:
    config = load_json_yaml(config_path)
    archive_path = Path(config["selected_archive"])
    if not archive_path.is_file():
        raise FileNotFoundError(f"selected Zenodo archive is missing: {archive_path}")
    cache_path = Path(config["checksum_policy"]["cache"])
    if not cache_path.is_absolute():
        cache_path = config_path.resolve().parents[2] / cache_path
    checksums = hash_archive_once(archive_path, cache_path)
    expected = config["identity"]

    members: list[dict[str, Any]] = []
    roles: Counter[str] = Counter()
    parse_states: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path, "r") as archive:
        bad_member = archive.testzip() if archive_path.stat().st_size < 2_000_000_000 else None
        readme = archive.read("README/README.md").decode("utf-8")
        for info in archive.infolist():
            role, ambiguity = _member_role(info.filename)
            parse_status, parse_detail = _parse_status(archive, info)
            roles[role] += 1
            parse_states[parse_status] += 1
            members.append(
                {
                    "relative_path": info.filename,
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "checksum": None if info.is_dir() else f"{info.CRC:08x}",
                    "checksum_algorithm": None if info.is_dir() else "zip_crc32",
                    "file_type": _file_type(info.filename),
                    "inferred_role": role,
                    "parse_status": parse_status,
                    "parse_detail": parse_detail,
                    "ambiguity": ambiguity,
                }
            )
        aggregate = {
            "archive_members": len(archive.infolist()),
            "files": sum(not x.is_dir() for x in archive.infolist()),
            "directories": sum(x.is_dir() for x in archive.infolist()),
            "compressed_member_bytes": sum(x.compress_size for x in archive.infolist()),
            "uncompressed_member_bytes": sum(x.file_size for x in archive.infolist()),
        }

    identity_checks = {
        "filename": archive_path.name == expected["archive_name"],
        "bytes": checksums["bytes"] == expected["expected_bytes"],
        "md5": checksums["md5"] == expected["expected_md5"],
        "sha256": checksums["sha256"] == expected["observed_sha256"],
        "readme_title": "Reinforcement Learning Control of Quantum Error Correction" in readme,
        "readme_willow": "Willow superconducting processor" in readme,
    }
    result = {
        "schema_version": "google-zenodo-inventory.v3",
        "dataset_read_only": True,
        "identity_status": "POSITIVELY_IDENTIFIED_OFFICIAL_ZENODO_V2" if all(identity_checks.values()) else "IDENTITY_FAILED",
        "identity": {
            **OFFICIAL_RELEASE,
            "local_path": str(archive_path.resolve()).replace("\\", "/"),
            "identity_checks": identity_checks,
            "archive_checksums": checksums,
            "zip_integrity_full_decompression": "NOT_RUN_DUE_TO_14.9_GB_UNCOMPRESSED_SIZE",
            "zip_test_bad_member": bad_member,
            "metadata_provenance": "Zenodo record plus archive README plus exact published MD5",
        },
        "candidate_search": discover_release(config),
        "summary": {**aggregate, "roles": dict(roles), "parse_states": dict(parse_states)},
        "members": members,
    }
    _write_json(artifact_dir / "zenodo_inventory.json", result)
    md = [
        "# Zenodo v2 local release inventory",
        "",
        f"**Identity:** `{result['identity_status']}`",
        f"**Local archive:** `{result['identity']['local_path']}`",
        f"**DOI:** `{OFFICIAL_RELEASE['doi']}`; **version:** `{OFFICIAL_RELEASE['version']}`; **creator:** {OFFICIAL_RELEASE['creator']}",
        f"**MD5:** `{checksums['md5']}` (published match: `{identity_checks['md5']}`)",
        f"**SHA-256:** `{checksums['sha256']}`; checksum cache hit: `{checksums['cache_hit']}`",
        "",
        f"The ZIP contains {aggregate['archive_members']:,} members ({aggregate['files']:,} files, {aggregate['directories']:,} directories) and {aggregate['uncompressed_member_bytes']:,} uncompressed bytes.",
        "Every member is listed in the JSON artifact with ZIP CRC-32, sizes, role, parse state, and ambiguity. CRC-32 is an integrity fingerprint from the signed central directory, not a cryptographic content-authenticity claim; archive authenticity is established by the published MD5 plus SHA-256.",
        "",
        "## Member-role counts",
        "",
        "| Role | Count |",
        "|---|---:|",
    ]
    md.extend(f"| {key} | {value:,} |" for key, value in sorted(roles.items()))
    md.extend(["", "## Ambiguities", "", "The three `aq2-ens9` prediction members use an undocumented pathway alias and are retained in the inventory. No member was silently omitted.", ""])
    (artifact_dir / "zenodo_inventory.md").write_text("\n".join(md), encoding="utf-8")
    return result


def estimate_inventory_cost(config_path: Path) -> dict[str, str]:
    config = load_json_yaml(config_path)
    path = Path(config["selected_archive"])
    return {
        "estimated_runtime": "20-40 seconds on first run; under 5 seconds with checksum cache",
        "estimated_read": f"{path.stat().st_size / 1e9:.2f} GB first run; central directory only on cache hit",
        "estimated_storage": "approximately 3-6 MB for JSON/Markdown inventory and checksum cache",
    }
