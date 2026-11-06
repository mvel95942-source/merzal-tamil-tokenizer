# Methods

**Status: research artifact.** This repository documents a tokenization experiment. It is
released so the design, measurements and failure modes can be inspected and reproduced — not
as a production component. The claims below are bounded by the protocol in this document;
[§8](#8-threats-to-validity) lists what the numbers do *not* show.

---

## 1. Research question

Tamil orthography is an abugida. A visible letter is a consonant plus an optional dependent
vowel sign or *pulli* (virama), encoded as two code points:

```
ம  U+0BAE           consonant
ி  U+0BBF           dependent vowel sign
மி U+0BAE U+0BBF    one visible letter, two code points
```

Subword tokenizers trained on raw Unicode can split between the two, producing pieces that
render as a bare consonant with a dotted circle plus an orphan sign. The piece count goes
down, the letter is destroyed, and the model has to relearn the orthography.

**Question:** can a tokenizer be trained whose pieces can *never* land inside a Tamil letter,
without giving up compression on Tamil, and without losing any input on decode?

**Sub-questions:**

1. Does pre-segmenting aksharas into single code points remove letter-cutting entirely?
2. What vocabulary size is worth its parameters for Tamil, holding the corpus fixed?
3. How does such a tokenizer compare with production multilingual tokenizers on genuinely
   unseen Tamil text?
4. What does the design cost in speed, memory and engineering honesty?

## 2. Design

The method has one central move: **change the alphabet, then train normally.**

Instead of asking the trainer not to split letters (which a data-driven objective will do
whenever it saves a token), each Tamil grapheme cluster is replaced by a single Private Use
Area code point before the trainer ever sees the text. A cluster is now one symbol. No
subword algorithm can cut a symbol in half. The decoder reverses the substitution.

Everything else — Unigram, byte fallback, Metaspace, special tokens — is standard.

## 3. Corpus

Token statistics were learned from Tamil and English web text plus a small code mix. Sources,
with their licenses, are listed in [`NOTICE`](NOTICE) and in each package's `metadata.json`:

| source | register | license |
|---|---|---|
| `ai4bharat/sangraha` (`verified/tam`) | curated Tamil web | CC BY 4.0 |
| `ai4bharat/IndicCorpV2` (`ta`) | broad Tamil web | CC0 1.0 |
| `wikimedia/wikipedia` (`20231101.ta`) | encyclopedic Tamil | CC BY-SA 3.0 |
| `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | English | ODC-By |
| `google/dakshina` (romanized `ta`) | Tanglish | CC BY-SA 4.0 |
| `codeparrot/github-code-clean` | ~5% code | Apache 2.0 |

**What is not reproduced.** The exact shuffled shards used for training are not
redistributed — they are derivable from the sources above but not bit-identical to them.
`tamil/tokenizer_train.py` reproduces the *method* exactly; it does not promise the same
vocabulary, because subword induction is sensitive to document order and sampling. This is
the single largest reproducibility gap in the release, and it is why the artifact is shipped
with a hash rather than presented as regenerable from scratch.

## 4. Akshara segmentation and PUA encoding

Implemented in [`tamil/akshara.py`](tamil/akshara.py).

The alphabet is built programmatically, not by hand:

- consonants: `U+0B95–U+0BB9` with Unicode category `Lo` → 25
- independent vowels: `U+0B85–U+0B94` with category `Lo` → 12
- aytham `ஃ` → 1
- for each consonant: the bare consonant, the consonant + pulli, and the consonant plus each
  dependent vowel sign → 25 × (2 + 11) = 325 nominal

After de-duplication the table holds **312 unique units**, assigned `U+E000…U+E137` in sorted
order. Order is deterministic, so the mapping is a pure function of the Unicode version's
categories rather than of the corpus.

Two implementation details matter:

1. **Longest-first replacement.** The normalizer applies 312 `Replace` operations, sorted by
   descending source length, after NFC. Without this order a bare `ம` would match before
   `மி` and the sign would be left stranded.
2. **The regex is cluster-shaped**, `[க-ஹ][ா-்ௗ]?|[அ-ஔ]|ஃ`, so it consumes an optional
   vowel sign or pulli but never more. Two adjacent signs are not one akshara.

Consequences, stated plainly:

- The map covers **canonical** akshara: consonant + at most one sign. Tamil text containing
  unusual stacked sequences, ZWJ/ZWNJ combinations, or a sign without its consonant is not
  covered. Those sequences degrade to per-code-point encoding, which is where the measured
  0.0392% orphan pieces come from.
- The encoder is not injective over *arbitrary* Unicode: a literal `U+E000` in the input is
  indistinguishable from a mapped akshara. Documented as a known limitation.

## 5. Vocabulary induction

Implemented in [`tamil/tokenizer_train.py`](tamil/tokenizer_train.py). Pipeline, in order:

| stage | setting | reason |
|---|---|---|
| Normalizer | `NFC`, then 312 longest-first `Replace` | one code point per akshara |
| Pre-tokenizer | `Metaspace(prepend_scheme="never")`, `Digits(individual_digits=True)` | no implicit whitespace mutation; digits split individually |
| Model | `Unigram` | handles the large PUA alphabet better than BPE at this vocab size |
| Trainer | `vocab_size` minus 256 byte pieces and 10 whitespace pieces; `initial_alphabet` = the 312 PUA units; `max_piece_length=16` | guarantees every akshara is representable from step 0 |
| Specials | 32 tokens, handed to the trainer so ids are fixed | pipeline compatibility (`pack.py`, `sft.py`) |
| Byte pieces | 256 `<0xNN>` appended after training with a floor score | Unigram does not emit them; `byte_fallback` needs them |
| Whitespace pieces | 10 `AddedToken(normalized=False)`: newlines, 2/4/8/12/16-space runs, 1–3 tabs | one token per indent run instead of one per space |
| Decoder | `Replace("▁", " ")` → `ByteFallback` → `Fuse` → 312 reverse replacements | order matters: bytes must fuse before PUA is restored |
| Post-processor | none | no implicit BOS/EOS |

`prepend_scheme="never"` is a deliberate choice, and it is the difference between the two
releases: the 32K used `"always"`, which consumes the first space of a string as the prefix
marker, so `" x"` decoded as `"x"`. The 64K does not prepend, and encodes whitespace runs
explicitly instead.

**Why 64K.** The 32K and 64K were trained with the same pipeline and compared on the same
held-out corpus. 64K reduces tokens per Tamil word from 1.853 to 1.693 (8.6% fewer tokens,
or equivalently the 32K needs 9.5% more for the same text). The cost is 32,000 extra
embedding rows; whether that trade is worth it depends on the model, which is exactly the
kind of question this release does not answer.

## 6. Evaluation protocol

Implemented in [`scripts/validate_tokenizer_holdout.py`](scripts/validate_tokenizer_holdout.py).

### 6.1 Held-out corpus

`HuggingFaceFW/fineweb-2`, config `tam_Taml`, split `test` — a different pipeline from every
training source, so it is unlikely to overlap them even before filtering.

| step | value |
|---|---|
| parquet file | `data/tam_Taml/test/000_00000.parquet` |
| parquet sha256 | `bba0eb727be01a334ae38f1cee3275ce55fb7c9c7d1e4a20b0b8bad5509c96d1` |
| rows read | 20,131 |
| kept | 20,000 |
| rejected as not-Tamil | 131 |
| dropped as too short (<60 chars) | 0 |
| dropped as duplicate | 0 |
| max characters per document | 4,000 |
| minimum Tamil-character share | 0.5 |
| totals | 40,207,389 chars · 4,491,695 words · 108,603,006 bytes |

Documents are stripped before encoding. This is deliberate: it isolates fertility from
whitespace handling, but it means the round-trip result does **not** exercise leading
whitespace, which is the 32K's known defect. That defect is documented and tested separately.

The **Tamil-dominant subset** (15,700 documents, ≥95% Tamil-script characters) is reported
separately because a corpus-wide average mixes Tamil with Latin, digits and punctuation and
can flatter a tokenizer that is merely good at English.

### 6.2 Metrics

| metric | definition | direction |
|---|---|---|
| tokens/word | `sum(tokens) / sum(words)` over the corpus; words = `str.split()` | lower is better |
| 95% CI | 1,000 bootstrap resamples **of documents**, recomputing the ratio of sums each time, fixed seed | — |
| chars/token | `sum(chars) / sum(tokens)` | higher is better |
| bytes/token | `sum(utf8 bytes) / sum(tokens)` | higher is better |
| cut letters | fraction of tokens whose **first code point** is a dependent vowel sign or pulli | lower is better |
| round-trip | fraction of documents where `decode(encode(doc)) == doc` exactly | higher is better |
| byte fallback | fraction of emitted tokens that are `<0xNN>` | lower is better |
| `<unk>` | fraction of emitted tokens equal to `<unk>` | lower is better |

The CI resamples documents rather than tokens because tokens within a document are
correlated; a token-level bootstrap would produce intervals that are far too narrow.
Ratio-of-sums is used instead of a mean of per-document ratios so that long documents count
proportionally, which is what a context window actually sees.

### 6.3 Baselines

Every baseline runs through the same harness, on the same documents, with
`add_special_tokens=False`: `gpt-4o` (o200k), `llama-3`, `gemma-4` (262K), `qwen-2.5`, and
for internal comparison `ours-tok32k`. Two additional 2.5B-parameter Indic baselines were
measured during development; they are excluded from the published tables because they are
not drop-in general-purpose tokenizers, and their measurements are not part of this release.

Baselines are loaded as released, including their own normalizers. A tokenizer that applies
aggressive normalization can look better on fertility while silently discarding distinctions;
the round-trip and cut-letter columns exist to make that visible.

### 6.4 Results

| tokenizer | tokens/word | 95% CI | cut letters | round-trip |
|---|---|---|---|---|
| **Merzal 64K** | **1.693** | [1.690, 1.696] | **0.04%** | **100%** |
| Merzal 32K | 1.853 | [1.849, 1.857] | 0.05% | 100% |
| Gemma-4 (262K) | 2.357 | [2.353, 2.361] | 12.0% | 100% |
| GPT-4o (o200k) | 7.502 | [7.489, 7.515] | 41.5% | 100% |
| Qwen-2.5 | 9.711 | [9.694, 9.727] | 40.3% | 100% |
| Llama-3 | 11.855 | [11.833, 11.875] | 33.0% | 100% |

Tamil-dominant subset: 64K **1.670** [1.667, 1.673], 0.04% cuts, 100% round-trip.

Differences between our two releases and the four production tokenizers are far larger than
the confidence intervals, which are narrow because the corpus is large and the estimator is
deterministic. Narrow intervals here mean *precisely estimated on this corpus*, not
*generalizes to all Tamil*. See §8.

## 7. Robustness testing

Implemented in [`scripts/hard_test_tokenizer.py`](scripts/hard_test_tokenizer.py). 12 suites,
12,156 assertions, run against the shipped artifact:

| suite | what it asserts |
|---|---|
| structure | vocab size, special ids, normalizer/pre-tokenizer/decoder wiring, no post-processor |
| akshara atomicity | each of the 312 units is exactly one token, in NFC **and** NFD input |
| block coverage | every code point in `U+0B80–U+0BFF` round-trips standalone, after a consonant, after a space |
| malformed Tamil | orphan signs, doubled pulli, sign-first strings — no crash, no loss |
| digits & punctuation | individual digits, runs, mixed |
| Unicode | emoji, ZWJ sequences, flag sequences, bidi controls, combining marks |
| fuzz | 9,000 seeded random mixed-script strings, round-trip + idempotence |
| whitespace | leading/trailing runs, indentation, tabs, CRLF, code snippets |
| specials | injection attempts, known collisions |
| transformers | load, batch, save/reload, thread safety |
| offsets | documents that offsets are approximate rather than asserting exactness |
| determinism | repeated encode of the same input yields identical ids |

**Result:** PASS, 0 errors, 5 documented warnings. The warnings are published, not suppressed:
special tokens are reachable from plain text; 18 orphan dependent-sign pieces exist; 54
mixed Tamil+Latin pieces exist; raw PUA and literal `U+2581` are lossy; lone surrogates raise.

The fuzz suite uses a fixed seed so a failure is reproducible; it is a smoke test for
crashes and loss, not a proof of correctness over all inputs.

## 8. Threats to validity

Read this section before quoting any number from this repository.

1. **One corpus, one split, one config.** Every headline number comes from a single FineWeb-2
   `tam_Taml` test file. It is genuinely held out from training, but it is one register of
   web Tamil. Classical, legal, social-media and spoken Tamil are represented unevenly. No
   second independent corpus was run.
2. **Fertility is not quality.** Tokens per word measures sequence length for a fixed text.
   It does not measure perplexity, downstream accuracy, or training efficiency. A model
   trained on this tokenizer was not evaluated against one trained on a baseline, so no claim
   is made that 28% fewer tokens yields 28% better results. The correct statement is: *the
   same text occupies 28% fewer positions*.
3. **The cut-letter metric is a proxy.** It counts tokens that *begin* with a dependent sign.
   That is a necessary condition for cutting an akshara, not a sufficient one, and it does
   not count a Tamil letter emitted as several byte-fallback tokens — those are counted in
   the byte-fallback column instead. The two metrics must be read together.
4. **Byte fallback hides fragmentation.** 0.01% of tokens are byte pieces. Those inputs
   round-trip, but they are expensive and semantically opaque. A zero `<unk>` rate with
   non-zero byte fallback is not the same as full vocabulary coverage.
5. **Training corpus is not bit-reproducible.** Sources are public, shards are not. The
   pipeline is reproducible; the exact vocabulary is not. Treat the artifact as the object of
   study.
6. **Comparison conditions favour nobody in particular, but are not neutral.** All tokenizers
   ran in the same Python process through `transformers`, so harness overhead is shared, but
   each baseline keeps its own normalizer and its own conventions for what counts as a token.
7. **Speed numbers are single-machine.** One Windows box, 6C/12T, warm page cache, Python
   3.14.4, `tokenizers` 0.22.2. Medians of 3 runs for 1 MB inputs, 1 run for 10 MB. The
   1M-id memory figures are **extrapolated** from 310,898 measured tokens, not measured at
   1M. Do not rank tokenizers across machines using these numbers.
8. **Whitespace defect not exercised in the headline round-trip.** Held-out documents are
   stripped, so the 32K's leading-space loss does not appear in its 100% round-trip figure.
   It is disclosed and demonstrated separately.
9. **No human evaluation.** Nobody was asked whether the tokenizer's segmentation is
   linguistically sensible. Compression and safety were measured; usefulness was not.
10. **PUA is a real trade-off.** It buys atomicity at the cost of a mapping the tokenizer must
    apply and reverse, plus a collision surface for literal PUA text. Both are documented;
    neither is free.

## 9. What production use would require

Not done here, and not claimed:

- a second and third held-out corpus, at least one non-web;
- a downstream ablation: train two small models, identical except for the tokenizer, compare
  loss and task accuracy at equal compute;
- sanitization of untrusted input against the 32 special-token strings, and stripping of raw
  PUA and `U+2581`;
- exact offset alignment if token classification is intended;
- a decision on embedding cost: 64K rows versus 32K rows is a real parameter bill;
- long-run stability: a vocabulary frozen against a Unicode version, and a migration policy
  for the next one.

## 10. Reproducing the measurements

```bash
pip install -r requirements.txt

# structural + robustness: 12,156 checks, artifact only, ~49 s
python scripts/hard_test_tokenizer.py \
    --tok merzal-tamil-tokenizer --compare merzal-tamil-tokenizer

# held-out fertility with bootstrap CIs: downloads ~72 MB of FineWeb-2
python scripts/validate_tokenizer_holdout.py \
    --tok merzal-tamil-tokenizer --limit 20000 --baselines core

# speed and memory: run on an idle machine
python scripts/bench_tokenizer.py --tok merzal-tamil-tokenizer

# rebuild metadata.json, SHA256SUMS and the release zip
python scripts/package_tokenizer_release.py
```

Every shipped file is covered by `SHA256SUMS`, and the repository sets `.gitattributes` to
`* -text` so that line-ending normalization cannot alter an artifact between clone and
checkout. The published `tokenizer.json` sha256 is
`861ebf3409db01b3964686bbbf3a71f0b721cd88b571bcbdb3d67196397a25d8`; a clone that produces a
different hash is a clone that has been transformed in transit.

## 11. Data and ethics statement

- No corpus text is redistributed. The packages contain only a vocabulary, a normalizer, a
  decoder and configuration.
- Training corpora retain their own licenses (CC BY 4.0, CC0, CC BY-SA 3.0/4.0, ODC-By,
  Apache 2.0); attribution is in [`NOTICE`](NOTICE).
- The evaluation corpus, FineWeb-2, is ODC-By 1.0 and is downloaded at evaluation time, not
  vendored.
- Tamil is a low-resource language for tokenizer coverage. The motivation for this work is
  that general-purpose tokenizers spend 12–42% of their Tamil tokens on fragments of letters;
  the experiment tests whether that is avoidable. It does not claim to fix Tamil language
  modelling, and the tokenizer carries no linguistic annotation beyond Unicode categories.
- Merzal is not affiliated with or endorsed by AI4Bharat, Google, Hugging Face or the
  Wikimedia Foundation.
