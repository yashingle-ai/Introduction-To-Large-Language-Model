# Assignment 3 — BPE and WordPiece on IndicCorp v2 (Hindi)

One script, `assignment3.py`, does the whole assignment.

```
pip install requests tokenizers
python assignment3.py
```

## What it does

| Step | Task | Output |
|:--|:--|:--|
| 1 | Download Hindi text from IndicCorp v2 and **sentence-tokenize** every paragraph | `data/sentences.txt` |
| 2 | Train / dev / test split, dev and test = 1000 sentences each, **length-stratified** | `data/train.txt`, `dev.txt`, `test.txt` |
| 3 | Train **BPE** and **WordPiece**, tokenize dev and test | `outputs/tokenized/`, `outputs/tokenizers/` |
| 4 | 4 training sizes × 3 vocabulary sizes × 2 algorithms = 24 runs | `outputs/RESULTS.md`, `outputs/results.json` |

Training sizes: 100,000 / 300,000 / 500,000 / 1,000,000 sentences.
Vocabulary sizes: 20,000 / 30,000 / 50,000.

## Design notes

**Why byte ranges instead of a download.** `data/hi-1.txt` is 26.7 GB. The script
issues HTTP `Range` requests for a handful of 20 MB windows spread evenly across
the file and reads only those. Windows are scattered rather than taken from the
head, because the head of these crawl dumps is typically a single news domain.

**Sentence tokenization.** Each corpus row is a paragraph. Sentences are split on
danda `।`, double danda `॥`, `?`, `!` and `.` followed by whitespace. Sentences are
NFC-normalized, de-duplicated, and kept only if they are 20–600 characters, have at
least 3 words, and are ≥80% Devanagari by letter (this drops menus, ad slugs and
English lines that sit inside the Hindi file).

**How dev and test avoid similar lengths (task 1 hint).** Sentences are bucketed
into 10 word-count bins (3–5, 6–8, 9–11, 12–15, 16–19, 20–24, 25–29, 30–39, 40–59,
60+). Dev takes 100 sentences from each bin and test takes 100 more, so both sets
are flat across the length range instead of clustering around the corpus mean.
`RESULTS.md` prints the per-bin counts and the min/median/max word counts of each
split as evidence.

**Keeping the two algorithms comparable.** BPE and WordPiece share the same
whitespace + punctuation pre-tokenizer, the same `min_frequency=2`, the same `##`
continuation marker, and are both trained only on the training split, so the
algorithm is the only thing left varying. The trainers come from the Hugging Face
`tokenizers` library (Rust), which is what makes 24 runs over a million sentences
finish in under two minutes.

| | BPE | WordPiece |
|:--|:--|:--|
| textbook training criterion | most **frequent** pair, `count(ab)` | best **likelihood gain**, `count(ab) / (count(a) × count(b))` |
| how a word is segmented | replay the learned merge list, in order | scan greedily, take the **longest** matching vocabulary piece |

**A caveat the measurements surfaced.** The report finds the two vocabularies are
~100% identical, which the first row above would not predict. The cause is the
library: huggingface's `WordPieceTrainer.train()` does not implement the likelihood
criterion — it trains an ordinary frequency-based BPE model and converts it with
`WordPiece::from_bpe()` (`tokenizers/src/models/wordpiece/trainer.rs`). So both runs
select their vocabulary with the *same* algorithm, and what this assignment actually
compares is the second row: one shared inventory, two segmentation strategies. That
is still a real, measurable difference — it changes ~20–37% of test sentences — but
it is not the training-criterion comparison the textbook describes, and the report
says so rather than claiming a result it did not produce.

**What is measured.** For dev and test, per configuration:

- tokens per sentence
- **tokens per word (fertility)** — the headline number, how many subword pieces one
  whitespace word costs
- tokens per word *excluding punctuation* — the pre-tokenizer splits punctuation off
  as its own "word" and punctuation is always exactly one token, so the plain figure
  is pulled toward 1; this column is the honest one
- characters per token
- % of words kept whole (tokenizer emitted exactly one token for the word)
- % continuation (`##`) tokens, % `[UNK]` tokens, distinct token types used
- vocabulary actually built vs. vocabulary requested

**Why averages alone are not enough.** Fertility *saturates*: past ~100k sentences,
the average tokens-per-word stops moving, which makes the task-3 table look like a
null result. It is not — the tokenizers genuinely differ, the average just hides it.
So the script also measures, directly:

- **vocabulary overlap** (Jaccard) between tokenizers trained at different sizes
- **tokenization agreement** — the % of test sentences that come out with the exact
  same token sequence under two settings
- **auto-selected examples** — a hand-picked sentence of common words comes out
  identical under every setting and demonstrates nothing, so the report picks the
  test sentences that actually changed the most, ranked by how much they changed

These are what turn "the numbers didn't move" into a measurable answer.

## Files

```
assignment3.py                    the whole assignment
run.log                           console log of the run
data/sentences.txt                all harvested sentences
data/train.txt dev.txt test.txt   the three splits
outputs/RESULTS.md                the report
outputs/results.json              every number, machine readable
outputs/tokenized/<tag>.dev.txt   tokenized dev set, one sentence per line
outputs/tokenized/<tag>.test.txt  tokenized test set
outputs/tokenizers/<tag>.json     the trained tokenizer
```

`<tag>` is `{bpe|wordpiece}_train{N}_vocab{V}`.

`outputs/RESULTS.md` and `results.json` are rewritten after every one of the 24
configurations, so the report is always complete for whatever has finished.
