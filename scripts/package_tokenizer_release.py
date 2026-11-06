"""Assemble the Merzal Tamil Tokenizer release from the artifact + reports.

Idempotent: safe to re-run after any report is refreshed.

  python scripts/package_tokenizer_release.py
  python scripts/package_tokenizer_release.py --version 1.0.0

Produces, under release/merzal-tamil-tokenizer/:
  tokenizer.json, metadata.json, SHA256SUMS, benchmarks/*
and a reproducible zip at release/merzal-tamil-tokenizer-v<version>.zip.
`README.md`, `LICENSE`, `NOTICE`, `CHANGES.md`, `VERSION`, `publish.py` and the
two tokenizer config files are authored by hand and only checked here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_FILES = [
    "tokenizer_hard_test.md", "tokenizer_hard_test.json",
    "tokenizer_holdout.md", "tokenizer_holdout.json",
    "tokenizer_bench.md", "tokenizer_bench.json",
    "fertility_v2.md", "fertility_v2.json",
]
# The headline table compares production general-purpose tokenizers. The two
# 2.5B-parameter-indic baselines are excluded from the shipped package; the full
# record, including those rows, stays in the private project's reports/ directory.
HIDE_IN_HOLDOUT = ("agaram-2.5b", "indicbert-v2")
HAND_AUTHORED = [
    "README.md", "LICENSE", "NOTICE", "CHANGES.md", "VERSION", "publish.py",
    "tokenizer_config.json", "special_tokens_map.json",
]
# Never include these in SHA256SUMS / the zip.
EXCLUDE = {"SHA256SUMS"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def reports_dir() -> Path:
    """Works in both the source project (reports/) and the standalone repo."""
    for cand in (ROOT / "reports", ROOT / "benchmarks",
                 ROOT / "merzal-tamil-tokenizer" / "benchmarks"):
        if cand.is_dir():
            return cand
    return ROOT / "reports"


def build_metadata(release: Path, src: Path, version: str) -> dict:
    spec = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    model = spec["model"]
    added = spec.get("added_tokens") or []
    specials = [a["content"] for a in added if a.get("special")]
    rdir = reports_dir()
    hard = load(rdir / "tokenizer_hard_test.json")
    hold = load(rdir / "tokenizer_holdout.json")
    bench = load(rdir / "tokenizer_bench.json")

    meta: dict = {
        "name": "merzal-tamil-tokenizer",
        "version": version,
        "release_date": date.today().isoformat(),
        "artifact": "tokenizer.json",
        "sha256": sha256(src / "tokenizer.json"),
        "vocab_size": len(spec["model"]["vocab"]) + len(
            [a for a in added if a["content"] not in {p for p, _ in model["vocab"]}]),
        "model_vocab_size": len(model["vocab"]),
        "added_tokens": len(added),
        "special_tokens": len(specials),
        "byte_fallback": bool(model.get("byte_fallback")),
        "model_type": model.get("type"),
        "max_length": 32768,
        "license": "Apache-2.0",
        "languages": ["ta", "en", "tanglish"],
        "normalizer": "NFC + 1 code point per Tamil akshara (312 units, PUA U+E000+)",
        "pre_tokenizer": "Metaspace(prepend_scheme=never, split) + Digits(individual_digits)",
        "training_sources": [
            "ai4bharat/sangraha (verified/tam, CC-BY-4.0)",
            "ai4bharat/IndicCorpV2 (ta, CC0-1.0)",
            "wikimedia/wikipedia (20231101.ta, CC-BY-SA-3.0)",
            "HuggingFaceFW/fineweb-edu (sample-10BT, ODC-By)",
            "google/dakshina (ta romanized, CC-BY-SA-4.0)",
            "codeparrot/github-code-clean (Apache-2.0, ~5% code)",
        ],
    }

    if hard:
        meta["hard_test"] = {
            "verdict": hard.get("verdict"),
            "checks": hard.get("checks"),
            "errors": hard.get("errors"),
            "warnings": hard.get("warnings"),
        }
    if hold:
        ours = hold["results"].get("ours-tok64k", {}).get("all")
        if ours:
            meta["heldout_validation"] = {
                "dataset": hold["meta"]["dataset"],
                "config": hold["meta"]["config"],
                "split": hold["meta"]["split"],
                "docs": ours["docs"],
                "tokens_per_word": ours["tokens_per_word"],
                "tokens_per_word_ci95": ours["tokens_per_word_ci95"],
                "cut_letter_rate": ours["cut_letter_rate"],
                "roundtrip_exact": ours["roundtrip_exact"],
            }
    if bench:
        b = bench["benchmarks"].get("tok64k")
        if b:
            t = b["throughput"].get("tamil_1mb", {})
            meta["performance"] = {
                "encode_mb_s_tamil": t.get("encode_mb_s"),
                "decode_mb_s_tamil": t.get("decode_mb_s"),
                "encode_tok_s_tamil": t.get("encode_tok_s"),
                "load_rss_delta_mb": b["load"]["rss_delta_mb"],
                "streaming_peak_delta_mb": b["streaming"]["peak_delta_mb"],
            }
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "artifacts" / "tok64k")
    ap.add_argument("--release", type=Path, default=ROOT / "release" / "merzal-tamil-tokenizer")
    ap.add_argument("--version", default=None, help="defaults to the VERSION file")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    rel = args.release
    rel.mkdir(parents=True, exist_ok=True)
    (rel / "benchmarks").mkdir(exist_ok=True)

    version = args.version or (rel / "VERSION").read_text(encoding="utf-8").strip()
    missing = [f for f in HAND_AUTHORED if not (rel / f).is_file()]
    if missing:
        print(f"ERROR: hand-authored files missing in {rel}: {missing}", file=sys.stderr)
        return 2

    # 1. artifact
    src_tok = args.src / "tokenizer.json"
    if not src_tok.is_file():
        print(f"ERROR: {src_tok} not found", file=sys.stderr)
        return 2
    dst_tok = rel / "tokenizer.json"
    dst_tok.write_bytes(src_tok.read_bytes())
    print(f"tokenizer.json  {dst_tok.stat().st_size:,} B  sha256 {sha256(dst_tok)}")

    # 2. reports
    reports = reports_dir()
    copied = []
    scrub_md = {"tokenizer_holdout.md": HIDE_IN_HOLDOUT, "fertility_v2.md": HIDE_IN_HOLDOUT}
    scrub_json = {"tokenizer_holdout.json": ("results", HIDE_IN_HOLDOUT),
                  "fertility_v2.json": (None, HIDE_IN_HOLDOUT)}
    for name in REPORT_FILES:
        p = reports / name
        if p.is_file():
            if name in scrub_md:
                hidden = scrub_md[name]
                kept = [ln for ln in p.read_text(encoding="utf-8").splitlines()
                        if not any(f"| {h} |" in ln for h in hidden)]
                (rel / "benchmarks" / name).write_text("\n".join(kept) + "\n", encoding="utf-8")
            elif name in scrub_json:
                doc = json.loads(p.read_text(encoding="utf-8"))
                container, hidden = scrub_json[name]
                target = doc[container] if container else doc
                for h in hidden:
                    target.pop(h, None)
                (rel / "benchmarks" / name).write_text(
                    json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            else:
                (rel / "benchmarks" / name).write_bytes(p.read_bytes())
            copied.append(name)
    print(f"benchmarks/     {len(copied)} report file(s)")

    # 3. metadata
    meta = build_metadata(rel, args.src, version)
    (rel / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    print(f"metadata.json   version {version}")

    # 4. checksums over everything shipped (except SHA256SUMS itself)
    files = sorted(p for p in rel.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts
                   and p.suffix != ".pyc" and p.name not in EXCLUDE)
    lines = [f"{sha256(p)}  {p.relative_to(rel).as_posix()}" for p in files]
    # LF on every platform: `sha256sum -c` treats the trailing CR as part of the name.
    (rel / "SHA256SUMS").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"SHA256SUMS      {len(lines)} file(s)")

    # 5. zip
    if not args.no_zip:
        out = ROOT / "release" / f"merzal-tamil-tokenizer-v{version}.zip"
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in files + [rel / "SHA256SUMS"]:
                z.write(p, arcname=f"merzal-tamil-tokenizer/{p.relative_to(rel).as_posix()}")
        print(f"zip             {out}  ({out.stat().st_size / 1024:.1f} KiB)")
        whole = hashlib.sha256(out.read_bytes()).hexdigest()
        out.with_suffix(".zip.sha256").write_text(f"{whole}  {out.name}\n", encoding="utf-8")
        print(f"zip sha256      {whole}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
