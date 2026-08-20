# Assignment 2 — TF and IDF Normalization

Reuses the dataset built in Assignment 1 (23 IndicCorpV2 languages, 1000 sentences
each, stratified 80/10/10) and compares **normalized vs. unnormalized TF and IDF**
vectors for a logistic-regression language identifier.

Everything lives in **one file**: [`assignment2.py`](assignment2.py).

## What is implemented

**1. Two TF normalizations**

| variant | formula | task |
|:--|:--|:--|
| `raw` | `tf = f` | unnormalized baseline (Assignment 1) |
| `length` | `tf = f / (total terms in the sentence)` | 1(a) |
| `maxfreq` | `tf = f / (count of the most frequent term in the sentence)` | 1(b) |

**2. One IDF normalization**

| variant | formula | task |
|:--|:--|:--|
| `raw` | `idf = log(N / df(t))` | unnormalized |
| `norm` | `idf = log(df_max / df(t))` | 2 |

`df_max` is the document frequency of the most frequent term across all sentences.
Since `df_max >= df(t)` for every term, this IDF is always `>= 0`.

**3. All 6 combinations** (3 TF × 2 IDF) are trained with identical
hyperparameters and compared by macro-F1.

## The one thing to point out when explaining this

**L2 row normalization is deliberately switched off.**

Both TF normalizations divide an entire row by a single per-sentence constant —
the sentence length, or its maximum count. L2 normalization *also* divides a row
by a per-row constant. Applying it afterwards would cancel both normalizations
exactly, and all three TF variants would produce the **identical** matrix.

So the comparison the assignment asks for is only meaningful with L2 off. That
is why `norm="l2"` from Assignment 1 does not appear here.

## Structure of `assignment2.py`

The file reads top to bottom in eight numbered sections:

| section | what it does |
|--:|:--|
| 1 | tokenisation — word 1/2-grams + char 2/3/4-grams |
| 2 | vocabulary (fitted on **train only**) and the raw count matrix |
| 3 | `normalize_tf()` — the two TF normalizations |
| 4 | `compute_idf()` — the IDF normalization |
| 5 | `SoftmaxRegression` — multinomial logistic regression, mini-batch Adam |
| 6 | metrics — confusion matrix, precision/recall/F1, macro-F1 |
| 7 | data loading |
| 8 | main loop over the 6 combinations, then writes the results file |

### Two implementation details worth knowing

- **Why not `\w` for tokenising?** In Python, `\w` does not match Indic combining
  vowel signs (Unicode category `Mc`/`Mn`), so `\w+` splits words like `किताबों`
  in the middle. Instead we map every punctuation/symbol/space character to a
  space and split on whitespace, which keeps Indic words intact.
- **Why Adam and not plain gradient descent?** The six feature matrices sit on
  very different scales — raw counts reach into the tens, length-normalized
  values are around 1/300. A single fixed learning rate cannot train all six
  fairly. Adam rescales each coordinate by its own gradient history, so one
  setting is fair to every variant.

## No sklearn

The vectorizer, the classifier and the metrics are written from scratch.
`scipy.sparse` is used only as a container for the sparse matrix and its matrix
product — no `sklearn` class is used anywhere.

## Run it

```powershell
python assignment2.py
```

Optional flags:

```powershell
python assignment2.py --epochs 30 --learning-rate 0.05 --min-df 2
```

By default it reads the splits from
`../Language_Identification_IndicCorpV2/outputs/data/`; override with `--data-dir`.

## Output

Everything is consolidated into one file:

- **`outputs/RESULTS.md`** — setup, formulas, the 6-combination table, the
  marginal effect of each normalization, per-language scores for the best
  combination, confusions, training curves and sample predictions
- `outputs/results.json` — the same numbers, machine-readable
