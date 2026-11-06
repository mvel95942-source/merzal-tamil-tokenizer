# Merzal Tamil Tokenizer

**A Tamil tokenizer that keeps Tamil letters whole.**

Tamil is written with dependent vowel signs and a *pulli* (virama) that modify the preceding
consonant. Byte-level BPE tokenizers routinely cut between the consonant and its sign, so one
visible letter becomes two meaningless pieces. This tokenizer maps every Tamil grapheme
cluster (*akshara*) to a single code point before segmentation, so a learned piece cannot land
inside a letter — and it still round-trips **every** input byte-for-byte.

Two artifacts ship here: the **64K (v1.0.0)** and the legacy **32K (v0.1.0)**.

**Hugging Face:** [Vel04/merzal-tamil-tokenizer](https://huggingface.co/Vel04/merzal-tamil-tokenizer)
(64K) · [Vel04/merzal-tamil-tokenizer-32k](https://huggingface.co/Vel04/merzal-tamil-tokenizer-32k)
(legacy) — both public, both hash-verified against this repository.

> **Status: research artifact.** This is a documented tokenization experiment, published so
> the design, measurements and failure modes can be inspected and reproduced. It is **not a
> production-ready component**: a single held-out corpus, no downstream ablation, no human
> evaluation. The protocol, its threats to validity and what production use would still
> require are in **[METHODS.md](METHODS.md)**. If you quote a number from here, quote its
> scope with it.

---

## Headline results

Measured on **20,000 Tamil documents the tokenizer never saw**:
[FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) `tam_Taml` / `test`,
40.2M characters · 4.49M words · 108.6 MB, disjoint from every training source.
Fertility is `sum(tokens) / sum(words)`; the 95% CI resamples documents.

| tokenizer | tokens/word ↓ | 95% CI | cut letters ↓ | exact round-trip |
|---|---|---|---|---|
| **Merzal Tamil Tokenizer (64K)** | **1.693** | **[1.690, 1.696]** | **0.04%** | **100%** |
| Merzal Tamil Tokenizer 32K *(legacy)* | 1.853 | [1.849, 1.857] | 0.05% | 100% |
| Gemma-4 (262K) | 2.357 | [2.353, 2.361] | 12.0% | 100% |
| GPT-4o (o200k) | 7.502 | [7.489, 7.515] | 41.5% | 100% |
| Qwen-2.5 | 9.711 | [9.694, 9.727] | 40.3% | 100% |
| Llama-3 | 11.855 | [11.833, 11.875] | 33.0% | 100% |

On the Tamil-dominant subset (≥95% Tamil script, 15,700 documents) the 64K reaches
**1.670** tokens/word [1.667, 1.673] with **0.04%** cut letters and **100%** round-trip.

## What we achieved

- **28% fewer tokens than Gemma-4**, **4.4× fewer than GPT-4o**, **5.7× fewer than Qwen-2.5**
  and **7.0× fewer than Llama-3** on identical Tamil text — 28% more Tamil per context window.
  On this corpus alone that is ~**3.0 million fewer tokens** than Gemma-4 would need.
- **0.04% cut letters — 1 in 2,551.** The production tokenizers split **12.0%** (Gemma-4),
  **33.0%** (Llama-3), **40.3%** (Qwen-2.5) and **41.5%** (GPT-4o) of Tamil letters. We cut
  **300× fewer** letters than Gemma-4 and **1,000× fewer** than GPT-4o.
- **100% exact round-trip** on all 20,000 held-out documents, with **0.0000% `<unk>`** and
  only 0.01% byte fallback. Nothing is unrepresentable.
- **8.6% fewer tokens for the 64K upgrade** than our own 32K (1.853 → 1.693 tokens/word;
  the 32K needs 9.5% more for the same text).
- **12,156-check stress test passes** — 312 akshara units atomic in NFC *and* NFD, every code
  point in `U+0B80–U+0BFF`, malformed Tamil, 9,000 seeded fuzz strings, emoji ZWJ, bidi
  controls, CRLF and code indentation. **0 errors.**
- **Whitespace-exact.** Leading, trailing and repeated whitespace survive; code and
  indentation round-trip. (The 32K v1 dropped a leading space — fixed in v1.0.0.)
- **Small and cheap to load:** 108 ms load, **+70 MB RSS** for the whole 63,995-piece model;
  streaming 100 MB grows RSS by **2.8 MB**.

## How we got here

The method is one move with a lot of care around it: **change the alphabet, then train
normally.** Each of the 312 canonical Tamil akshara is mapped to a single Private Use Area
code point before the trainer sees any text, so a subword algorithm cannot cut a symbol in
half; the decoder reverses the mapping. Everything else is standard Unigram with byte
fallback.

| stage | what was done |
|---|---|
| 1. Corpus | Tamil web (Sangraha, IndicCorpV2, Wikipedia), English (FineWeb-Edu), Tanglish (Dakshina), ~5% code — sources and licenses in [`NOTICE`](NOTICE) |
| 2. Akshara table | built programmatically from Unicode categories: 25 consonants + 12 vowels + aytham, each with pulli/sign forms → **312 unique units**, `U+E000…U+E137` |
| 3. Normalizer | NFC, then 312 `Replace` ops applied **longest-first** so `ம` cannot match before `மி` |
| 4. Model | `Unigram`, `initial_alphabet` = the 312 PUA units, `max_piece_length=16`, `Metaspace(prepend_scheme="never")` |
| 5. Closing the alphabet | 256 `<0xNN>` byte pieces appended post-training (Unigram does not emit them), plus 10 whitespace-run tokens |
| 6. Iteration | 32K v1 shipped, then failed its own whitespace test (`prepend_scheme="always"` ate a leading space) → 64K v2 fixed it and cut fertility 1.853 → 1.693 |
| 7. Evaluation | 20,000 unseen FineWeb-2 `tam_Taml` documents, ratio-of-sums fertility, document-level bootstrap CIs, cut-letter and round-trip audits |
| 8. Stress | 12 suites, 12,156 assertions: 312 units atomic in NFC *and* NFD, all of `U+0B80–U+0BFF`, malformed Tamil, 9,000 fuzz strings |
| 9. Performance | load time, per-domain throughput, batch, 200k-char document, 100 MB streaming RSS, id memory |
| 10. Packaging | hash every file, `* -text` in `.gitattributes` so no clone can silently alter an artifact |

Full protocol, metric definitions, and the ten threats to validity:
**[METHODS.md](METHODS.md)**.

## The two tokenizers

| | **64K — v1.0.0** | **32K — v0.1.0 (legacy)** |
|---|---|---|
| Directory | [`merzal-tamil-tokenizer/`](merzal-tamil-tokenizer/) | [`merzal-tamil-tokenizer-32k/`](merzal-tamil-tokenizer-32k/) |
| Vocab | 63,995 | 32,000 |
| Model | Unigram, byte fallback, NFC | Unigram, byte fallback, NFC |
| Tokens/word (held-out) | **1.693** | 1.853 |
| Whitespace | exact | drops one leading space |
| Status | **recommended** | kept for reproducibility |

The 32K is the first-generation tokenizer, published because it is the tokenizer that packed
our 350M Tamil pretraining run. Use the 64K unless you need to reproduce that pipeline.
Neither is the tokenizer of the Merzal 1 model (which uses Gemma's tokenizer); this is a
standalone tokenizer release.

## Quick start

```python
from transformers import AutoTokenizer

# from the Hub
tok = AutoTokenizer.from_pretrained("Vel04/merzal-tamil-tokenizer")
# or from a clone
# tok = AutoTokenizer.from_pretrained("merzal-tamil-tokenizer")

text = "வணக்கம் நண்பா, how are you? 123"
ids = tok(text)["input_ids"]
assert tok.decode(ids) == text
```

Raw `tokenizers`, without `transformers`:

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("merzal-tamil-tokenizer/tokenizer.json")
enc = tok.encode("தமிழ் மொழி", add_special_tokens=False)
assert tok.decode(enc.ids) == "தமிழ் மொழி"
```

Whitespace and code are safe:

```python
s = "def f():\n    return 'தமிழ்'\n"
assert tok.decode(tok.encode(s, add_special_tokens=False).ids) == s
```

## How it works

| Stage | Setting |
|---|---|
| Normalizer | NFC, then one `Replace` per akshara: **312 Tamil clusters → `U+E000…U+E137`** |
| Pre-tokenizer | `Metaspace(prepend_scheme="never", split=True)` + `Digits(individual_digits=True)` |
| Model | Unigram, 63,990 pieces, `byte_fallback=true`, `unk_id=3` |
| Decoder | `▁` → space, byte fallback, fuse, 312 PUA → Tamil replacements |
| Added tokens | 32 special/tool/reserved + 10 whitespace runs (`"  "`, `"    "`, tabs, newlines, indent runs) |
| Post-processor | none — no implicit BOS/EOS |

Anything outside the Tamil block falls back to a 256-piece byte alphabet, so arbitrary
Unicode — emoji, CJK, rare scripts — always encodes and always round-trips. `<unk>` never
fires.

Core ids are fixed for the training pipeline:
`<pad>=0`, `<s>=1`, `</s>=2`, `<unk>=3`, `<|user|>=4`, `<|assistant|>=5`, plus tool/chat,
`<think>`, FIM and 16 reserved tokens.

## Speed and memory

Release artifact, warm tokenizer, single process, CPU only — Windows 11, 6C/12T, 16.4 GB RAM,
Python 3.14.4, `tokenizers` 0.22.2. MB = 10⁶ UTF-8 bytes.

| workload | encode MB/s | encode tok/s | decode MB/s | round-trip |
|---|---|---|---|---|
| Tamil 1 MB | 0.98 | 67,723 | 1.89 | yes |
| Tamil 10 MB | 0.65 | 44,716 | 1.77 | yes |
| Tanglish 1 MB | 0.47 | 161,502 | 1.44 | yes |
| English 1 MB | 0.47 | 96,881 | 1.60 | yes |
| Code 1 MB | 0.13 | 52,710 | 1.39 | yes |
| Digits 1 MB | 0.33 | 296,900 | 1.06 | yes |
| Emoji 1 MB | 0.52 | 240,877 | 1.29 | yes |

- **Load:** 108 ms, **+70 MB RSS** for the full vocabulary.
- **Long document:** 200,000 Tamil characters → 776 ms encode, 294 ms decode.
- **Batch:** 584,645 tokens/s, 2,506 docs/s via `encode_batch`.
- **Streaming 100 MB:** 0.64 MB/s at **+2.8 MB** peak RSS — the encoder does not accumulate.
- **Id memory:** 1M ids = **27.2 MB** as a Python list, **2.0 MB** as `uint16` (max id 63,994),
  4.0 MB as `int32`.

These are single-machine numbers, not a ranking against other tokenizers measured elsewhere.

## Reproduce

```bash
pip install -r requirements.txt

# 1. 12-suite structural / robustness test (artifact only, ~49 s)
python scripts/hard_test_tokenizer.py --tok merzal-tamil-tokenizer --compare merzal-tamil-tokenizer

# 2. held-out fertility with bootstrap CIs (downloads FineWeb-2 tam_Taml test, ~72 MB)
python scripts/validate_tokenizer_holdout.py --tok merzal-tamil-tokenizer --limit 20000 --baselines core

# 3. speed and memory on an idle machine
python scripts/bench_tokenizer.py --tok merzal-tamil-tokenizer

# 4. rebuild metadata.json, SHA256SUMS and the release zip
python scripts/package_tokenizer_release.py
```

Benchmark scripts add the repo root to `sys.path`; run them from the repository root.

## Repository layout

```
METHODS.md                      research protocol, metrics, threats to validity
merzal-tamil-tokenizer/         64K v1.0.0 package (Hugging Face-ready)
  tokenizer.json                the artifact, 63,995 pieces
  tokenizer_config.json         PreTrainedTokenizerFast, model_max_length=32768
  special_tokens_map.json       core + 28 additional special tokens
  metadata.json                 hashes, vocab stats, measured results
  benchmarks/                   hard test, held-out validation, speed/memory, fertility
  README.md                     model card
  CHANGES.md  NOTICE  LICENSE    versioning, data attribution, Apache-2.0
  publish.py                    dry-run, private-by-default Hugging Face upload
  LAUNCH.md                     release and verification checklist
merzal-tamil-tokenizer-32k/     legacy 32K v0.1.0 package
benchmarks/                     raw JSON + markdown for every shipped result
scripts/
  hard_test_tokenizer.py        12-suite stress test (12,156 checks)
  validate_tokenizer_holdout.py independent-corpus fertility with bootstrap CI
  bench_tokenizer.py            encode/decode throughput, RSS, streaming, id memory
  package_tokenizer_release.py  rebuilds metadata, checksums and the zip
  package_tokenizer_32k_release.py
tamil/                          akshara table, trainer, fertility harness, source list
```

## Known limitations

Read these before using the tokenizer for anything beyond plain encoding — they are
documented, not hidden.

1. **Special tokens are matched inside plain text.** All 32 special strings are reachable
   from raw input and are dropped on decode. Sanitize untrusted text, or assert that no
   special id appears when encoding a corpus.
2. **Raw Private Use Area characters collide with the akshara alphabet.** Literal `U+E000+`
   decodes to a Tamil letter and literal `U+2581` decodes to a space. Strip both upstream.
3. **Character offsets are approximate.** The `Replace` normalizer aligns offsets to the last
   code point of a cluster; `return_offsets_mapping=True` needs re-alignment for
   token classification.
4. **18 orphan pieces** begin with a dependent sign. They can only match malformed Tamil and
   waste a negligible number of slots.
5. **Lone UTF-16 surrogates raise** instead of encoding; input must be valid UTF-8.
6. **0.04% of letters become orphan pieces** on real-world text (1 in ~2,551), concentrated in
   malformed sequences. All 312 canonical akshara units are atomic by construction.

## License and attribution

- Tokenizer code and vocabulary: **Apache License 2.0** — see [`LICENSE`](LICENSE).
- Training corpora keep their own licenses (CC BY 4.0, CC0, CC BY-SA 3.0/4.0, ODC-By,
  Apache 2.0). No corpus text is redistributed; attribution is in [`NOTICE`](NOTICE).
- Validation corpus: FineWeb-2 by Hugging Face, ODC-By 1.0.
- Merzal is not affiliated with or endorsed by AI4Bharat, Google, Hugging Face or the
  Wikimedia Foundation. Provided AS IS, without warranty (Apache 2.0, section 7).

## Citation

```bibtex
@misc{merzal_tamil_tokenizer_2026,
  title  = {Merzal Tamil Tokenizer: an akshara-safe, lossless Tamil tokenizer},
  author = {Merzal},
  year   = {2026},
  note   = {64K v1.0.0; 32K v0.1.0 legacy},
  url    = {https://github.com/mvel95942-source/merzal-tamil-tokenizer}
}
```
