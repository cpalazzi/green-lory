#!/usr/bin/env python3
"""Fetch and checksum-verify the seven files in the paper's public code deposit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse

FILES = {"COPYING.txt", "main.py", "p_constraints.py", "p_driver.py",
         "p_optimisation_parent.py", "p_select_locations.py", "p_toolbox.py"}
URL = "https://data.mendeley.com/public-api/datasets/v4yz7778mh/files?folder_id=root&version=1"


def download(url):
    return subprocess.run(["curl", "-fLsS", "--max-time", "25", url], check=True, capture_output=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = download(URL)
    records = json.loads(metadata)
    if {record["filename"] for record in records} != FILES or len(records) != len(FILES):
        raise ValueError("Public deposit inventory differs from the expected seven files")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "file_metadata.json").write_bytes(metadata)
    for record in records:
        details = record["content_details"]
        url = details["download_url"]
        if urlparse(url).hostname != "data.mendeley.com":
            raise ValueError("Unexpected public download host")
        content = download(url)
        sha = hashlib.sha256(content).hexdigest()
        if sha != details["sha256_hash"]:
            raise ValueError(f"Checksum mismatch: {record['filename']}")
        (args.output / record["filename"]).write_bytes(content)
        print(record["filename"], sha, flush=True)


if __name__ == "__main__":
    main()
