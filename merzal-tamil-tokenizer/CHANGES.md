# Changelog

All notable changes to the Merzal Tamil Tokenizer are recorded here.
This project follows [Semantic Versioning](https://semver.org/).

## 1.0.0 — 2026-11-06

First public-ready release of the **Merzal Tamil Tokenizer**. Artifact: `tokenizer.json`,
vocab 63,995.

### Added
- 64K Unigram tokenizer for Tamil script, Tanglish and English, with a
  byte-fallback alphabet so arbitrary Unicode always round-trips.
- Akshara-safe normalization: every Tamil grapheme cluster is mapped to one
  Private Use Area code point before segmentation, so a learned piece cannot
  cut a canonical akshara in half.
- First-class whitespace pieces (spaces, tabs, newlines, common indent runs) so
  code and indentation survive exactly.
- Chat/tool/reserved special tokens (32 total) and the fixed core ids
  `<pad>=0, <s>=1, </s>=2, <unk>=3, <|user|>=4, <|assistant|>=5`.
- Release package: `tokenizer_config.json` with `model_max_length=32768`,
  `special_tokens_map.json`, model card, NOTICE with data attribution,
  benchmark reports and `SHA256SUMS`.

### Changed from the 32K tokenizer (v0.x)
- Vocabulary 32,000 -> 63,995.
- Tamil fertility on the project benchmark improved from **2.259** to **1.975**
  tokens per word (12.6% fewer tokens); on the independent FineWeb-2 held-out
  corpus it improved from **1.853** to **1.693** (8.6% fewer).
- Whitespace handling fixed. The 32K model used `Metaspace(prepend_scheme="always")`,
  which silently dropped a leading space (`" x"` -> `"x"`). This release uses
  `prepend_scheme="never"` and dedicated whitespace pieces; leading and repeated
  whitespace now round-trip exactly.
- Broader coverage: digits, punctuation runs, emoji ZWJ sequences, Indic
  numerals and mixed Tamil/Latin text all round-trip.

### Known limitations
- The 32 special/tool/reserved tokens are matched inside plain text; inputs
  must be sanitized or encoded with `add_special_tokens=False` and a check that
  no special id appears.
- Raw Private Use Area characters that overlap the akshara alphabet
  (`U+E000`+), and literal `U+2581`, are lossy: they decode to the mapped Tamil
  letter or a space. Sanitize both upstream.
- Character offset mappings are approximate: the `Replace` normalizer aligns to
  the last code point, so `return_offsets_mapping` should not be used for
  token-level labeling without re-alignment.
- 18 vocabulary pieces begin with a dependent vowel sign or pulli. They can only
  match malformed input and waste a negligible number of slots.
- A lone UTF-16 surrogate raises instead of encoding; text must be valid UTF-8.

See `benchmarks/` for the hard-test, held-out validation and speed/memory
reports that back these claims.
