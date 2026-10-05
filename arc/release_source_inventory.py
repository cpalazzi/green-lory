#!/usr/bin/env python3
"""Write or verify a deterministic inventory of a staged source release.

The inventory is intentionally limited to Git-visible source files. Large
ignored datasets and generated campaign artifacts are identified separately by
run manifests and are not part of this source-tree digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_EXCLUDED_PREFIXES = ("results/",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_visible_paths(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    decoded = [
        raw.decode("utf-8", errors="surrogateescape")
        for raw in completed.stdout.split(b"\0")
        if raw
    ]
    return sorted(set(decoded))


def _is_excluded(relative: str, excluded_prefixes: tuple[str, ...]) -> bool:
    normalized = relative.replace(os.sep, "/").lstrip("./")
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in excluded_prefixes
    )


def _file_entry(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if path.is_symlink():
        return {
            "path": relative,
            "type": "symlink",
            "target": os.readlink(path),
        }
    if not path.is_file():
        raise ValueError(f"Inventory source is not a regular file: {relative}")
    return {
        "path": relative,
        "type": "file",
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _tree_sha256(entries: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8", errors="surrogateescape")
    return hashlib.sha256(canonical).hexdigest()


def build_inventory(
    root: Path,
    *,
    excluded_prefixes: tuple[str, ...] = DEFAULT_EXCLUDED_PREFIXES,
) -> dict[str, Any]:
    root = root.resolve()
    paths = [
        relative
        for relative in _git_visible_paths(root)
        if not _is_excluded(relative, excluded_prefixes)
    ]
    entries = [_file_entry(root, relative) for relative in paths]
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "git_cached_and_untracked_nonignored_source",
        "excluded_prefixes": list(excluded_prefixes),
        "entry_count": len(entries),
        "tree_sha256": _tree_sha256(entries),
        "entries": entries,
    }


def write_inventory(
    root: Path,
    output: Path,
    *,
    excluded_prefixes: tuple[str, ...] = DEFAULT_EXCLUDED_PREFIXES,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite source inventory: {output}")
    payload = build_inventory(root, excluded_prefixes=excluded_prefixes)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return payload


def verify_inventory(root: Path, inventory_path: Path) -> dict[str, Any]:
    root = root.resolve()
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported source-inventory schema: {payload.get('schema_version')!r}"
        )
    expected_entries = payload.get("entries")
    if not isinstance(expected_entries, list) or not expected_entries:
        raise ValueError("Source inventory contains no entries")

    actual_entries = [_file_entry(root, str(entry["path"])) for entry in expected_entries]
    if actual_entries != expected_entries:
        for expected, actual in zip(expected_entries, actual_entries):
            if expected != actual:
                raise ValueError(
                    "Staged source differs from inventory at "
                    f"{expected.get('path')}: expected={expected}, actual={actual}"
                )
        raise ValueError("Staged source differs from inventory")

    expected_tree = str(payload.get("tree_sha256", ""))
    actual_tree = _tree_sha256(actual_entries)
    if actual_tree != expected_tree:
        raise ValueError(
            f"Source tree digest mismatch: expected {expected_tree}, found {actual_tree}"
        )
    if int(payload.get("entry_count", -1)) != len(actual_entries):
        raise ValueError("Source inventory entry count is inconsistent")
    return {
        "status": "passed",
        "entry_count": len(actual_entries),
        "tree_sha256": actual_tree,
        "inventory_sha256": _sha256(inventory_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write", help="Inventory local Git-visible source")
    write.add_argument("--root", type=Path, required=True)
    write.add_argument("--output", type=Path, required=True)
    write.add_argument(
        "--exclude-prefix",
        action="append",
        default=list(DEFAULT_EXCLUDED_PREFIXES),
        help="Repository-relative prefix to omit; repeatable",
    )

    verify = subparsers.add_parser("verify", help="Verify a staged release")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--inventory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "write":
        payload = write_inventory(
            args.root,
            args.output,
            excluded_prefixes=tuple(args.exclude_prefix),
        )
        summary = {
            "status": "written",
            "entry_count": payload["entry_count"],
            "tree_sha256": payload["tree_sha256"],
            "output": str(args.output.resolve()),
        }
    else:
        summary = verify_inventory(args.root, args.inventory)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - concise CLI failure
        print(f"Source inventory validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
