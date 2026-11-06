# Launch checklist — Merzal Tamil Tokenizer v1.0.0

Status: **package complete, private publish ready**. Flipping the repo public is the
owner's decision; nothing here uploads by default.

Everything in this folder is reproducible from the repo:

| Step | Command |
|---|---|
| Rebuild package + zip + checksums | `python scripts/package_tokenizer_release.py` |
| Verify files and hashes | `python merzal-tamil-tokenizer/publish.py --repo Vel04/merzal-tamil-tokenizer --check` |
| Dry-run upload (default) | `python merzal-tamil-tokenizer/publish.py --repo Vel04/merzal-tamil-tokenizer` |
| Private upload | `python merzal-tamil-tokenizer/publish.py --repo Vel04/merzal-tamil-tokenizer --upload` |
| Public upload | `python merzal-tamil-tokenizer/publish.py --repo Vel04/merzal-tamil-tokenizer --upload --public` |

## 1. Pre-flight (30 minutes, once)

1. **Checksums.** `publish.py --check` must print `OK: N files verified against SHA256SUMS`
   and the recorded sha256 of `tokenizer.json`.
2. **Hard test.** `python scripts/hard_test_tokenizer.py --tok artifacts/tok64k --compare artifacts/tok32k`
   must end `PASS (with warnings)` with 0 errors. The warnings are documented in
   `benchmarks/tokenizer_hard_test.md` and in the model card's limitations.
3. **Held-out validation.** `benchmarks/tokenizer_holdout.md` must show 20,000 FineWeb-2
   `tam_Taml` `test` documents, 0.0392% cut letters (1 in ~2,551) and 100% exact round-trip
for `ours-tok64k`.
4. **Speed/memory.** `benchmarks/tokenizer_bench.md` must be from a quiet machine (close
   browsers, no training jobs). Re-run if it was measured under load.
5. **Config sanity.** Load the release folder with `transformers` 4.x *and* 5.x once:

   ```python
   from transformers import AutoTokenizer
   t = AutoTokenizer.from_pretrained("merzal-tamil-tokenizer")
   assert t.model_max_length == 32768
   assert len(t.all_special_tokens) == 32
   assert t.convert_tokens_to_ids("<|assistant|>") == 5
   assert t.decode(t("வணக்கம் நண்பா")["input_ids"]) == "வணக்கம் நண்பா"
   ```

6. **Nothing stale.** `benchmarks/` must match `reports/` exactly (the packaging script
   copies them, so just re-run it).

## 2. Private publish

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
python merzal-tamil-tokenizer/publish.py --repo Vel04/merzal-tamil-tokenizer --check
python merzal-tamil-tokenizer/publish.py --repo Vel04/merzal-tamil-tokenizer          # dry run
python merzal-tamil-tokenizer/publish.py --repo Vel04/merzal-tamil-tokenizer --upload # private
```

Then open the repo and confirm: repo is **private**, the model card renders, the file
list has `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `LICENSE`,
`NOTICE`, `CHANGES.md`, `SHA256SUMS`, `metadata.json`, `benchmarks/`.

## 3. Post-upload verification (from a clean environment)

```bash
pip install -U tokenizers transformers
python - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("Vel04/merzal-tamil-tokenizer")
s = "வணக்கம் நண்பா, how are you? 123"
ids = t(s)["input_ids"]
print(ids[:12], t.decode(ids))
assert t.decode(ids) == s
print("vocab", len(t), "specials", len(t.all_special_tokens))
PY
```

Also test the raw library path, which is the one the training code uses:

```python
from tokenizers import Tokenizer
t = Tokenizer.from_file("tokenizer.json")
assert t.decode(t.encode("தமிழ் மொழி").ids) == "தமிழ் மொழி"
```

## 4. Public flip

Only after the private repo has been loaded by one other machine and the card has been
read end to end.

```powershell
python merzal-tamil-tokenizer/publish.py --repo Vel04/merzal-tamil-tokenizer --upload --public
```

## 5. Announcement — claims that are backed by evidence

Quote the numbers from `benchmarks/`, not from memory.

- **Lossless.** 100% exact round-trip on 20,000 unseen Tamil documents; the 32K v1
  silently dropped leading whitespace, this release does not.
- **Akshara-safe.** 0.0392% of letters (1 in ~2,551) fall into an orphan piece, across the
  full `U+0B80–U+0BFF` block, malformed input, and 9,000 fuzz strings. All 312 canonical
  akshara units are atomic by construction. Comparable tokenizers cut 12–42% of letters
  (see `benchmarks/tokenizer_holdout.md`).
- **Compact.** 1.693 tokens per Tamil word on FineWeb-2 `test`, versus 2.357 for the
  Gemma-4 tokenizer and 7.502 for GPT-4o's o200k on the same corpus.
- **Fast enough.** See `benchmarks/tokenizer_bench.md` for encode/decode MB/s, batch
  throughput and RSS on this machine. Do not claim a speed ranking against other
  tokenizers unless they were measured in the same run.
- **Honest limits.** Special tokens are matched inside plain text; raw PUA and literal
  `U+2581` are lossy; char offsets are approximate. Say so before someone else finds it.

Suggested one-liner (fill the speed figure from the bench report):

> A 64K Tamil tokenizer that keeps Tamil letters whole: 1.693 tokens/word on unseen Tamil
> (0.04% orphan pieces), 100% round-trip, byte fallback for everything else. Apache-2.0,
> weights-free. 28% fewer tokens than Gemma-4.

## 6. Rollback

- Make the repo private again in the HF UI, or
- `python -c "from huggingface_hub import HfApi; HfApi().delete_repo('Vel04/merzal-tamil-tokenizer', repo_type='model')"`
  (irreversible; prefer flipping to private).

## 7. Known gaps (fine for v1.0.0, fix before a big campaign)

- **Character offsets are wrong** for token classification. The `Replace` normalizer
  aligns offsets to the last code point of a cluster. Needs a custom offset post-pass.
- **No sanitizer helper.** Ship a small `sanitize(text)` (strip PUA `U+E000–U+E0FF`,
  literal `U+2581`, lone surrogates, and optionally escape special-token lookalikes)
  with the next release.
- **Orphan pieces.** 18 vocabulary entries start with a dependent sign; they only match
  malformed input. A future retrain can drop them.
- **No upstream model uses this tokenizer yet.** Merzal 1 ships the Gemma tokenizer;
  say "tokenizer release", not "the Merzal model's tokenizer".
