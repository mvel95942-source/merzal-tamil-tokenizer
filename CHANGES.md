# Changelog

All notable changes to the **Merzal Tamil Tokenizer** are recorded here. Each released
package also carries its own `CHANGES.md` / `metadata.json` next to the artifact.

## Repository — 2026-11-06

- Framed the release as a **research artifact**: added [`METHODS.md`](METHODS.md) with the
  full protocol, metric definitions, threats to validity and production-readiness gap, and a
  status banner in every README and model card.
- Published both artifacts: the 64K tokenizer (v1.0.0) and the legacy 32K tokenizer
  (v0.1.0), with full benchmark evidence and reproduction scripts.
- Added the independent held-out validation on FineWeb-2 `tam_Taml` / `test`
  (20,000 documents, 4.49M words) with bootstrap 95% confidence intervals.
- Added the speed and memory benchmark: load cost, per-domain throughput, batch, long
  documents, 100 MB streaming RSS and token-id memory.
- Fixed the cut-letter reporting precision. The headline figure is **0.0392%** (1 in
  ~2,551), not 0.0%; earlier drafts rounded it away.

## 64K — 1.0.0 / 2026-11-06

First public release of the Merzal Tamil Tokenizer. Artifact `tokenizer.json`, vocab
63,995. See [`merzal-tamil-tokenizer/CHANGES.md`](merzal-tamil-tokenizer/CHANGES.md).

- 64K Unigram model with a 256-piece byte-fallback alphabet.
- Akshara-safe normalization: 312 Tamil clusters mapped to `U+E000…U+E137` before
  segmentation, restored in the decoder.
- Whitespace-exact: `Metaspace(prepend_scheme="never")` plus dedicated whitespace pieces.
- 32 special/tool/reserved tokens with fixed core ids `<pad>=0 … <|assistant|>=5`.
- Held-out result: **1.693 tokens/word**, 0.04% cut letters, 100% exact round-trip.

## 32K — 0.1.0 (legacy) / 2026-11-06

First-generation 32,000-piece tokenizer, published for reproducibility of the 350M Tamil
pretraining run and superseded by the 64K release.

- Vocab 32,000, 6 special tokens, Unigram with byte fallback.
- **Known defect:** `Metaspace(prepend_scheme="always")` drops one leading space.
- Held-out result: **1.853 tokens/word**, 0.05% cut letters, 100% exact round-trip on
  stripped documents.
