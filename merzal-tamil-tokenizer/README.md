---
language:
- ta
- en
license: apache-2.0
library_name: tokenizers
tags:
- tamil
- tokenizer
- unigram
- sentencepiece
- akshara
- byte-fallback
- merzal
---

# Merzal Tamil Tokenizer

**64K vocabulary · akshara-safe · lossless.** A Unigram tokenizer for Tamil script,
Tanglish and English that **keeps all 312 canonical Tamil akshara atomic** and
**round-trips every input exactly**.

Tamil is written with dependent vowel signs and a pulli (virama) that modify the
preceding consonant. Byte-level BPE tokenizers routinely cut between the consonant and
its sign, which turns a single visible letter into two meaningless pieces. This
tokenizer maps every Tamil grapheme cluster (akshara) to one Private Use Area code point
before segmentation, so a learned piece cannot land mid-letter; the decoder maps them
back. Anything outside the Tamil block falls back to a 256-piece byte alphabet, so no
input is ever unknown.

**Highlights:** **1.693 tokens/word** on 20,000 unseen Tamil documents — **28% fewer
than Gemma-4**, 4.4× fewer than GPT-4o · **0.04% cut letters** (1 in 2,551) vs **12–41.5%**
for the big production tokenizers · **100% exact round-trip** · loads in 108 ms with
**+70 MB RSS**.

- **Vocab:** 63,995 (63,990 model pieces + 5 added whitespace tokens)
- **Model:** Unigram, byte fallback, NFC
- **License:** Apache-2.0 (tokenizer code) with data attribution in `NOTICE`
- **Version:** 1.0.0
- **SHA256:** `861ebf3409db01b3964686bbbf3a71f0b721cd88b571bcbdb3d67196397a25d8`

## Quick start

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("YOUR-ORG/merzal-tamil-tokenizer")
text = "வணக்கம் நண்பா, how are you? 123"
ids = tok(text)["input_ids"]
print(ids)
print(tok.decode(ids))          # வணக்கம் நண்பா, how are you? 123
```

Raw `tokenizers`, without `transformers`:

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
enc = tok.encode("தமிழ் மொழி", add_special_tokens=False)
assert tok.decode(enc.ids) == "தமிழ் மொழி"
```

The tokenizer is whitespace-exact, so it is safe for code and indentation:

```python
s = "def f():\n    return 'தமிழ்'\n"
assert tok.decode(tok.encode(s, add_special_tokens=False).ids) == s
```

### Special tokens

| id | token | | id | token |
|---|---|---|---|---|
| 0 | `<pad>` | | 5 | `<|assistant|>` |
| 1 | `<s>` | | 6–15 | `<|system|>`, tool call/response, `<think>`, FIM |
| 2 | `</s>` | | 16–31 | `<|reserved_0|>` … `<|reserved_15|>` |
| 3 | `<unk>` | | | |
| 4 | `<|user|>` | | | |

Core ids 0–5 are fixed because the training pipeline (`pack.py`, `sft.py`) depends on
them. `add_special_tokens=False` is the safe default for corpus encoding.

## Design

| Stage | Setting |
|---|---|
| Normalizer | NFC, then one `Replace` per akshara: 312 Tamil clusters → `U+E000…U+E137` |
| Pre-tokenizer | `Metaspace(prepend_scheme="never", split=True)` + `Digits(individual_digits=True)` |
| Model | Unigram, 63,990 pieces, `byte_fallback=true`, `unk_id=3` |
| Decoder | `▁` → space, byte fallback, fuse, 312 PUA → Tamil replacements |
| Added tokens | 32 special/tool/reserved + 10 whitespace runs (`"  "`, `"    "`, tabs, newlines, indent runs) |
| Post-processor | none (no implicit BOS/EOS) |

`prepend_scheme="never"` is the important v2 fix: the 32K v1 used `"always"`, which
silently dropped a leading space (`" x"` decoded as `"x"`). This release preserves
leading, trailing and repeated whitespace exactly.

## Training data

Token statistics were learned from Tamil and English web text plus a small code mix.
No text is redistributed with this package.

| Dataset | License | Share |
|---|---|---|
| [ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) (`verified/tam`) | CC BY 4.0 | Tamil |
| [ai4bharat/IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2) (`ta`) | CC0 1.0 | Tamil |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.ta`) | CC BY-SA 3.0 | Tamil |
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) | ODC-By | English |
| [google/dakshina](https://huggingface.co/datasets/google/dakshina) (romanized `ta`) | CC BY-SA 4.0 | Tanglish |
| [codeparrot/github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean) | Apache 2.0 | ~5% code |

## Validation on an independent held-out corpus

Trained-on data above is disjoint from the evaluation corpus: **FineWeb-2
`tam_Taml`, split `test`** ([HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2),
ODC-By), 20,000 documents · 40.2M characters · 4.49M words · 108.6 MB that never appeared
in training. Fertility is `sum(tokens) / sum(words)`; the 95% CI resamples documents.

| tokenizer | tokens/word | 95% CI | cut letters | exact round-trip |
|---|---|---|---|---|
| **Merzal Tamil Tokenizer (64K)** | **1.693** | **[1.690, 1.696]** | **0.04%** | **100%** |
| Merzal Tamil Tokenizer 32K (v1) | 1.853 | [1.849, 1.857] | 0.05% | 100% |
| gemma-4 (262k) | 2.357 | [2.353, 2.361] | 12.0% | 100% |
| gpt-4o (o200k) | 7.502 | [7.489, 7.515] | 41.5% | 100% |
| qwen-2.5 | 9.711 | [9.694, 9.727] | 40.3% | 100% |
| llama-3 | 11.855 | [11.833, 11.875] | 33.0% | 100% |

**The achievements, in one block:**

- **1.693 tokens per Tamil word** — **28% fewer tokens than Gemma-4**, **4.4× fewer than
  GPT-4o**, **5.7× fewer than Qwen-2.5** and **7.0× fewer than Llama-3** on the same text.
  That is roughly **3.0 million fewer tokens than Gemma-4** across this 108.6 MB corpus:
  28% more Tamil per context window, 28% less compute per token budget.
- **0.04% cut letters — 1 in 2,551.** The competitors split 12.0% (Gemma-4), 33.0%
  (Llama-3), 40.3% (Qwen-2.5) and 41.5% (GPT-4o) of Tamil letters, so this release cuts
  **300× fewer** letters than Gemma-4 and **1,000× fewer** than GPT-4o.
- **100% exact round-trip** on all 20,000 documents, with **0.0000% `<unk>`** and only
  0.01% byte fallback.
- **8.6% fewer tokens than our own 32K v1** (the 32K needs 9.5% more for the same text) —
  the 64K vocabulary pays for itself.
- Tamil-dominant subset (≥95% script, 15,700 docs): **1.670** tokens/word
  [1.667, 1.673], still **0.04% cut letters** and **100% round-trip**.

Full aggregates for every tokenizer in this table are in
`benchmarks/tokenizer_holdout.json`.

## Speed and memory

Measured on the release artifact with a warm tokenizer. Full method and the JSON
results are in `benchmarks/tokenizer_bench.md`.

| workload | encode MB/s | encode tok/s | decode MB/s | round-trip |
|---|---|---|---|---|
| Tamil 1 MB | 0.98 | 67,723 | 1.89 | yes |
| Tamil 10 MB | 0.65 | 44,716 | 1.77 | yes |
| Tanglish 1 MB | 0.47 | 161,502 | 1.44 | yes |
| English 1 MB | 0.47 | 96,881 | 1.60 | yes |
| Code 1 MB | 0.13 | 52,710 | 1.39 | yes |
| Digits 1 MB | 0.33 | 296,900 | 1.06 | yes |
| Emoji 1 MB | 0.52 | 240,877 | 1.29 | yes |

On this 6-core / 12-thread Windows machine (Python 3.14.4, `tokenizers` 0.22.2):

- **Load:** 108 ms, **+70 MB RSS** for the whole 63,995-piece model.
- **Long document:** a 200,000-character Tamil document encodes in **776 ms** and decodes
  in **294 ms**.
- **Batch:** **584,645 tokens/s** and 2,506 docs/s through `encode_batch`.
- **Streaming:** 100 MB processed at 0.64 MB/s with a peak RSS increase of only
  **2.8 MB** — the encoder does not accumulate the corpus.
- **Token-id memory:** 1M ids cost **27.2 MB as a Python list**, **2.0 MB as `uint16`**
  (the vocab fits in 16 bits: max id 63,994).

## Robustness

`benchmarks/tokenizer_hard_test.md` runs 12,156 checks over the real artifact:

- every one of the 312 akshara units is exactly one token, in NFC and NFD form;
- every code point in `U+0B80–U+0BFF` round-trips standalone, after a consonant and
  after a space, with no `<unk>`;
- 9,000 seeded mixed-script fuzz strings round-trip and re-encode idempotently;
- indent runs, tabs, CRLF, code snippets, emoji ZWJ sequences and bidi controls survive;
- 0 cut letters across all 312 akshara units, 0 unknown tokens; verdict **PASS**
  (5 documented warnings). The separate 20,000-document held-out run measures
  **0.0392%** orphan pieces overall, concentrated in malformed sequences.

## Known limitations

Read these before using the tokenizer for anything but plain encoding.

1. **Special tokens are matched inside plain text.** All 32 special/tool/reserved strings
   are reachable from raw input, and decoding drops them. Sanitize untrusted text or
   assert `not (set(ids) & SPECIAL_IDS)` when encoding a corpus.
2. **Raw PUA collides with the akshara alphabet.** Literal `U+E000`-range characters
   decode to Tamil letters, and a literal `U+2581` decodes to a space. Strip both
   upstream. (Six known lossy inputs are listed in the hard-test report.)
3. **Character offsets are approximate.** The `Replace` normalizer aligns offsets to the
   last code point of a cluster, so `return_offsets_mapping=True` is not suitable for
   token classification without re-alignment.
4. **18 orphan pieces** begin with a dependent sign and can only match malformed Tamil.
   They are harmless but waste a negligible number of slots.
5. **Lone UTF-16 surrogates raise.** Text must be valid UTF-8; this is intentional.
6. **No model ships with this tokenizer yet.** Merzal 1 uses the Gemma tokenizer. This is
   a standalone tokenizer release, not "the Merzal model's tokenizer".

## Files

| File | Purpose |
|---|---|
| `tokenizer.json` | the artifact (load with `tokenizers` or `transformers`) |
| `tokenizer_config.json` | `PreTrainedTokenizerFast`, `model_max_length=32768` |
| `special_tokens_map.json` | core + 28 additional special tokens |
| `metadata.json` | machine-readable versions, hashes, vocab stats, results |
| `benchmarks/` | hard test, held-out validation, speed/memory, 2k-doc fertility |
| `CHANGES.md` | what changed from the 32K v1 and why |
| `NOTICE` | dataset attribution and CC/ODC compliance |
| `SHA256SUMS` | checksums for every shipped file |
| `LAUNCH.md` | release and verification checklist |

## License and attribution

- Tokenizer code and vocabulary: Apache License 2.0. See `LICENSE`.
- Training corpora keep their own licenses; attribution is in `NOTICE`. No corpus text
  is included in this package.
- Validation corpus: FineWeb-2 by Hugging Face, ODC-By 1.0.
- Merzal is not affiliated with or endorsed by AI4Bharat, Google, Hugging Face or the
  Wikimedia Foundation. Provided AS IS, without warranty (Apache 2.0, section 7).
