#!/usr/bin/env python3
"""Write the mutable submission facts beside a byte-immutable run manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--job-id", action="append", required=True)
    parser.add_argument("--qa-job-id", required=True)
    parser.add_argument("--cluster")
    parser.add_argument("--mail-user")
    parser.add_argument("--mail-type")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite submission record: {output}")
    manifest = Path(args.manifest)
    payload = {
        "schema_version": 1,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "shard_job_ids": args.job_id,
        "merge_qa_job_id": args.qa_job_id,
        "cluster": args.cluster,
        "requested_mail_user": args.mail_user,
        "requested_mail_type": args.mail_type,
    }
    snapshots = []
    for job_id in [*args.job_id, args.qa_job_id]:
        command = ["scontrol"]
        if args.cluster:
            command += ["-M", args.cluster]
        command += ["--oneliner", "show", "job", job_id]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            snapshots.append({"job_id": job_id, "returncode": result.returncode,
                              "stdout": result.stdout.strip(), "stderr": result.stderr.strip()})
        except (OSError, subprocess.TimeoutExpired) as exc:
            snapshots.append({"job_id": job_id, "capture_error": str(exc)})
    payload["scheduler_snapshots"] = snapshots
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Submission record: {output}")


if __name__ == "__main__":
    main()
