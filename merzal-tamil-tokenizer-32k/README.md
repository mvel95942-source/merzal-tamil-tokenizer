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
- byte-fallback
- legacy
- merzal
---

# Merzal Tamil Tokenizer 32K (v1, legacy)

The first-generation 32,000-piece tokenizer. It is the tokenizer used to pack the 350M
Tamil pretraining run, and it is kept here for reproducibility. **It is superseded by the
64K Merzal Tamil Tokenizer**, which is smaller on text, whitespace-exact and otherwise
identical in design.

- **Vocab:** 32,000 · **Model:** Unigram, byte fallback, NFC
- **Version:** 0.1.0 (legacy) · **License:** Apache-2.0
- **SHA256:** `bbedd9c0e6b993f1b9b2505a8991219bae3539ad44794e92e86a3c1e0506ee07`

## Results on the same independent held-out corpus

FineWeb-2 `tam_Taml` / `test`, 20,000 documents, 40.2M characters, 4.49M words.

| tokenizer | tokens/word | 95% CI | cut letters | exact round-trip |
|---|---|---|---|---|
| Merzal Tamil Tokenizer (64K) | **1.693** | [1.690, 1.696] | 0.04% | 100% |
| **Merzal Tamil Tokenizer 32K (this)** | **1.853** | [1.849, 1.857] | 0.05% | 100% |
| gemma-4 (262k) | 2.357 | [2.353, 2.361] | 12.0% | 100% |

The 64K model needs **8.6% fewer tokens** for the same Tamil text (this 32K model needs
9.5% more). This 32K model is still
far ahead of general-purpose tokenizers (1.853 vs 2.357 for Gemma-4) and it cuts
**240× fewer** Tamil letters than Gemma-4 (0.05% vs 12.0%).

## Known defect: leading whitespace

This version uses `Metaspace(prepend_scheme="always")`. The first space of a string is
consumed by the prefix marker, so:

```python
tok.decode(tok.encode(" x").ids)   # "x"   <- the leading space is lost
tok.decode(tok.encode("  ").ids)   # " "   <- a leading run collapses to one space
```

Interior and trailing whitespace are preserved (`"a  b"` round-trips). The held-out
documents are stripped before encoding, so the 100% round-trip above does not exercise
this path. If you need exact leading whitespace, use the 64K release.

## Usage

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("YOUR-ORG/merzal-tamil-tokenizer-32k")
ids = tok("தமிழ் மொழி")["input_ids"]
print(tok.decode(ids))
```

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
assert tok.decode(tok.encode("தமிழ் மொழி", add_special_tokens=False).ids) == "தமிழ் மொழி"
```

Special tokens: `<pad>=0`, `<s>=1`, `</s>=2`, `<unk>=3`, `<|user|>=4`, `<|assistant|>=5`.
No tool/chat/reserved tokens and no whitespace-run tokens.

## Files

| File | Purpose |
|---|---|
| `tokenizer.json` | the 32,000-piece artifact |
| `tokenizer_config.json` | `PreTrainedTokenizerFast`, `model_max_length=32768` |
| `special_tokens_map.json` | the six special tokens |
| `metadata.json` | hashes, vocab stats, benchmark summary |
| `SHA256SUMS` | checksums |

## License and attribution

Apache License 2.0 (`LICENSE`). Training data attribution is in `NOTICE`; the corpora are
the same as the 64K release and no corpus text is redistributed. Provided AS IS, without
warranty (Apache 2.0, section 7).
