#!/usr/bin/env python
"""Publish the Merzal Tamil Tokenizer package to Hugging Face.

Dry-run by default: nothing is downloaded, created or uploaded unless `--upload`
is passed. Uploads go to a PRIVATE repo unless `--public` is passed explicitly.

Run by the owner with their own HF login (`hf auth login` or HF_TOKEN env var).
This script never prints or stores tokens.

    python publish.py --repo YOUR-ORG/merzal-tamil-tokenizer            # dry run
    python publish.py --repo YOUR-ORG/merzal-tamil-tokenizer --check    # verify files+hashes
    python publish.py --repo YOUR-ORG/merzal-tamil-tokenizer --upload   # private upload
    python publish.py --repo YOUR-ORG/merzal-tamil-tokenizer --upload --public

The upload is an idempotent folder sync of this directory. `--check` verifies that
tokenizer.json matches the sha256 recorded in SHA256SUMS before anything leaves the
machine.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

REQUIRED = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "README.md",
    "LICENSE",
    "NOTICE",
    "CHANGES.md",
    "VERSION",
    "metadata.json",
    "SHA256SUMS",
]

# Never upload local scratch files.
IGNORE = ["*.pyc", "__pycache__", ".ipynb_checkpoints", "*.zip", ".DS_Store"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sums(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        if name:
            out[name.strip()] = digest.strip()
    return out


def check() -> int:
    missing = [f for f in REQUIRED if not (HERE / f).is_file()]
    if missing:
        print(f"ERROR: missing files: {missing}", file=sys.stderr)
        return 2
    sums = parse_sums(HERE / "SHA256SUMS")
    bad = []
    for name, digest in sums.items():
        p = HERE / name
        if not p.is_file():
            bad.append(f"{name}: listed but missing")
            continue
        got = sha256(p)
        if got != digest:
            bad.append(f"{name}: {got} != {digest}")
    listed = {n for n in sums}
    unlisted = [p.name for p in HERE.iterdir()
                if p.is_file() and p.name != "SHA256SUMS" and p.name not in listed]
    if bad:
        print("ERROR: checksum mismatch", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 3
    if unlisted:
        print(f"WARNING: files not in SHA256SUMS: {sorted(unlisted)}", file=sys.stderr)
    print(f"OK: {len(sums)} files verified against SHA256SUMS")
    print(f"     tokenizer.json {sha256(HERE / 'tokenizer.json')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="target repo id, e.g. YOUR-ORG/merzal-tamil-tokenizer")
    ap.add_argument("--upload", action="store_true", help="actually create the repo and upload (default: dry run)")
    ap.add_argument("--public", action="store_true", help="create a PUBLIC repo (default: private)")
    ap.add_argument("--check", action="store_true", help="verify required files and SHA256SUMS, then exit")
    ap.add_argument("--revision", default="main", help="branch/tag to upload to")
    args = ap.parse_args()

    rc = check()
    if rc != 0 or args.check:
        return rc

    files = sorted(p.relative_to(HERE).as_posix() for p in HERE.rglob("*")
                   if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts)
    size = sum((HERE / f).stat().st_size for f in files)
    visibility = "public" if args.public else "private"

    print(f"repo:       {args.repo}  ({visibility})")
    print(f"revision:   {args.revision}")
    print(f"folder:     {HERE}")
    print(f"files:      {len(files)} ({size / 1024:.1f} KiB)")
    for f in files:
        print(f"  {f:34s} {(HERE / f).stat().st_size:>10,d} B")

    if not args.upload:
        print("\n[dry-run] nothing created, nothing uploaded.")
        print("Pass --upload to publish, and --public to make the repo public.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()  # token comes from the local login / HF_TOKEN; never printed
    try:
        who = api.whoami()["name"]
    except Exception:
        print("ERROR: not logged in. Run `hf auth login` (or set HF_TOKEN) and retry.", file=sys.stderr)
        return 3
    print(f"\nLogged in as: {who}")
    print(f"Creating {visibility} repo {args.repo} (if needed) ...")
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    print("Uploading ...")
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(HERE),
        revision=args.revision,
        ignore_patterns=IGNORE,
        commit_message="Merzal Tamil Tokenizer v1.0.0",
    )
    print(f"Done. {visibility.capitalize()} repo: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
