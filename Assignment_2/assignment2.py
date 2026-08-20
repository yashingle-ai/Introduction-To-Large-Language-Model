"""
Assignment 2 -- TF and IDF normalization for Indic language identification.
===========================================================================

We reuse the dataset built in Assignment 1 (23 IndicCorpV2 languages,
1000 sentences each, split 80/10/10 with stratified sampling) and study how
*normalizing* the TF part and the IDF part of a TF-IDF vector changes the
accuracy of a logistic-regression language identifier.

Three TF variants x two IDF variants = the 6 required combinations.

    TF (f = raw count of term t in sentence d)
      1. "raw"     tf = f                        <- unnormalized (Assignment 1)
      2. "length"  tf = f / (total terms in d)   <- normalization (1a)
      3. "maxfreq" tf = f / (count of the most frequent term in d)   <- (1b)

    IDF (N = number of training sentences, df(t) = in how many it occurs)
      A. "raw"     idf = log(N / df(t))                  <- unnormalized
      B. "norm"    idf = log(df_max / df(t))             <- normalization (2)
                   where df_max = max over all terms of df(t), i.e. the
                   document frequency of the most frequent term across
                   sentences.

IMPORTANT DESIGN NOTE (worth explaining to the evaluator)
---------------------------------------------------------
We deliberately do NOT apply L2 row normalization here.  Both TF
normalizations divide an entire row by a single per-sentence constant
(the length, or the max count).  L2 normalization also divides a row by a
per-row constant, so it would cancel them exactly and all three TF variants
would collapse into the *same* matrix.  Turning L2 off is what makes this
comparison meaningful at all.

NO sklearn is used anywhere: the vectorizer, the classifier and the metrics
are all written from scratch.  `scipy.sparse` is used only as a storage
container for the sparse matrix and its matrix product.

Run:
    python assignment2.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

# The 6 required combinations: (TF variant, IDF variant).
COMBINATIONS = [
    ("raw", "raw"),
    ("raw", "norm"),
    ("length", "raw"),
    ("length", "norm"),
    ("maxfreq", "raw"),
    ("maxfreq", "norm"),
]

TF_DESCRIPTION = {
    "raw": "f (unnormalized raw count)",
    "length": "f / total number of terms in the sentence",
    "maxfreq": "f / count of the most frequent term in the sentence",
}
IDF_DESCRIPTION = {
    "raw": "log(N / df)  (unnormalized)",
    "norm": "log(df_max / df)  (normalized by the most frequent term across sentences)",
}



# 1.  TOKENISATION -- word 1/2-grams and character 2/3/4-grams


def _build_punctuation_table() -> dict[int, str]:
    """Map every punctuation / symbol / space / control char to a space.

    We cannot use the regex `\\w`, because in Python `\\w` does not match
    Indic combining vowel signs (category Mc/Mn), which would split words
    like "किताबों" in the middle.  Mapping punctuation to spaces and then
    splitting keeps every Indic word intact.
    """
    table: dict[int, str] = {}
    for codepoint in range(0x10000):
        if unicodedata.category(chr(codepoint))[0] in ("P", "S", "Z", "C"):
            table[codepoint] = " "
    return table


PUNCTUATION_TABLE = _build_punctuation_table()


def extract_features(sentence: str) -> list[str]:
    """Turn one sentence into the list of its n-gram terms (with repeats).

    Each term is prefixed with its family ("w1:", "w2:", "c2:", ...) so that
    a word and a character n-gram that happen to look the same stay distinct.
    """
    words = sentence.lower().translate(PUNCTUATION_TABLE).split()

    features = [f"w1:{w}" for w in words]
    features += [f"w2:{a}_{b}" for a, b in zip(words, words[1:])]

    padded = " " + " ".join(words) + " "
    for n in (2, 3, 4):
        features += [f"c{n}:{padded[i:i + n]}" for i in range(len(padded) - n + 1)]
    return features



# 2.  VOCABULARY AND THE RAW COUNT MATRIX


def build_vocabulary(documents: list[list[str]], min_df: int) -> tuple[dict[str, int], np.ndarray]:
    """Collect terms seen in at least `min_df` TRAINING sentences.

    Returns the term -> column index mapping and the document-frequency
    array aligned to those column indices.  Fitting on train only keeps
    validation/test statistics out of the representation.
    """
    document_frequency: Counter[str] = Counter()
    for terms in documents:
        document_frequency.update(set(terms))

    kept = sorted(term for term, count in document_frequency.items() if count >= min_df)
    vocabulary = {term: index for index, term in enumerate(kept)}
    df_array = np.array([document_frequency[term] for term in kept], dtype=np.float64)

    print(f"    terms seen           : {len(document_frequency):,}")
    print(f"    dropped (df < {min_df})    : {len(document_frequency) - len(kept):,}")
    print(f"    vocabulary size      : {len(kept):,}")
    return vocabulary, df_array


def build_count_matrix(documents: list[list[str]], vocabulary: dict[str, int]) -> csr_matrix:
    """Build the sparse matrix of RAW term counts (rows = sentences)."""
    indptr = [0]
    indices: list[int] = []
    values: list[float] = []

    for terms in documents:
        counts: Counter[int] = Counter()
        for term in terms:
            column = vocabulary.get(term)
            if column is not None:
                counts[column] += 1
        indices.extend(counts.keys())
        values.extend(counts.values())
        indptr.append(len(indices))

    return csr_matrix(
        (np.array(values, dtype=np.float64), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int64)),
        shape=(len(documents), len(vocabulary)),
    )



# 3.  THE TF NORMALIZATIONS


def normalize_tf(counts: csr_matrix, mode: str) -> csr_matrix:
    """Apply one of the three TF definitions to a raw count matrix."""
    if mode == "raw":
        return counts.copy()

    if mode == "length":
        # 1(a) total number of terms in the sentence = row sum of the counts
        divisor = np.asarray(counts.sum(axis=1)).ravel()
    elif mode == "maxfreq":
        # 1(b) count of the most frequent term in that sentence
        divisor = counts.max(axis=1).toarray().ravel()
    else:
        raise ValueError(f"unknown TF mode: {mode}")

    divisor = np.maximum(divisor, 1.0)  # guard against an empty sentence

    scaled = counts.copy()
    terms_per_row = np.diff(scaled.indptr)
    scaled.data = scaled.data / np.repeat(divisor, terms_per_row)
    return scaled



# 4.  THE IDF NORMALIZATION


def compute_idf(df_array: np.ndarray, n_documents: int, mode: str) -> np.ndarray:
    """Unnormalized log(N/df), or normalized log(df_max/df)."""
    if mode == "raw":
        return np.log(n_documents / df_array)
    if mode == "norm":
        # 2. normalize by the document frequency of the most frequent term.
        # df_max >= df(t) for every t, so this is always >= 0, and it is
        # exactly 0 for the single most frequent term.
        return np.log(df_array.max() / df_array)
    raise ValueError(f"unknown IDF mode: {mode}")


def apply_idf(tf_matrix: csr_matrix, idf: np.ndarray) -> csr_matrix:
    """Multiply every column j of the TF matrix by idf[j]."""
    weighted = tf_matrix.copy()
    weighted.data = weighted.data * idf[weighted.indices]
    return weighted



# 5.  LOGISTIC REGRESSION (multinomial / softmax), written from scratch


class SoftmaxRegression:
    """Multinomial logistic regression trained with mini-batch Adam.

    Model:      p(y=c | x) = softmax(x @ W + b)_c
    Loss:       mean cross-entropy + (l2/2) * ||W||^2
    Gradient:   dL/dW = X^T (P - Y) / B  +  l2 * W

    Adam is used instead of plain gradient descent because the six feature
    matrices live on wildly different scales (raw counts vs. values around
    1/300).  Adam rescales each coordinate by its own gradient history, so
    one learning rate trains all six variants fairly.
    """

    def __init__(self, n_features, n_classes, learning_rate=0.05, l2=1e-6,
                 epochs=20, batch_size=512, seed=42):
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        """Numerically stable softmax (subtract the row max before exp)."""
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def _scores(self, X: csr_matrix) -> np.ndarray:
        return X @ self.W + self.b

    def fit(self, X_train, y_train, X_val, y_val):
        rng = np.random.default_rng(self.seed)
        self.W = np.zeros((self.n_features, self.n_classes))
        self.b = np.zeros(self.n_classes)

        # Adam state
        mean_W = np.zeros_like(self.W)
        variance_W = np.zeros_like(self.W)
        mean_b = np.zeros_like(self.b)
        variance_b = np.zeros_like(self.b)
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        step = 0

        n_samples = X_train.shape[0]
        one_hot = np.zeros((n_samples, self.n_classes))
        one_hot[np.arange(n_samples), y_train] = 1.0

        best_val_f1, best_W, best_b = -1.0, self.W.copy(), self.b.copy()
        history = []

        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(n_samples)
            total_loss = 0.0

            for start in range(0, n_samples, self.batch_size):
                rows = order[start:start + self.batch_size]
                X_batch = X_train[rows]
                Y_batch = one_hot[rows]

                probabilities = self._softmax(self._scores(X_batch))
                total_loss += -np.log(probabilities[np.arange(len(rows)), y_train[rows]] + 1e-12).sum()

                error = (probabilities - Y_batch) / len(rows)
                gradient_W = X_batch.T @ error + self.l2 * self.W
                gradient_b = error.sum(axis=0)

                step += 1
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                mean_W = beta1 * mean_W + (1 - beta1) * gradient_W
                variance_W = beta2 * variance_W + (1 - beta2) * gradient_W ** 2
                self.W -= self.learning_rate * (mean_W / bias_correction1) / (
                    np.sqrt(variance_W / bias_correction2) + epsilon)

                mean_b = beta1 * mean_b + (1 - beta1) * gradient_b
                variance_b = beta2 * variance_b + (1 - beta2) * gradient_b ** 2
                self.b -= self.learning_rate * (mean_b / bias_correction1) / (
                    np.sqrt(variance_b / bias_correction2) + epsilon)

            val_f1 = macro_f1(y_val, self.predict(X_val), self.n_classes)
            history.append({"epoch": epoch,
                            "train_loss": total_loss / n_samples,
                            "val_macro_f1": val_f1})

            if val_f1 > best_val_f1:                       # keep the best model
                best_val_f1, best_W, best_b = val_f1, self.W.copy(), self.b.copy()

            if epoch % 5 == 0 or epoch == 1:
                print(f"      epoch {epoch:3d}  loss={total_loss / n_samples:.4f}  val macro-F1={val_f1:.4f}")

        self.W, self.b = best_W, best_b
        self.history = history
        self.best_val_f1 = best_val_f1
        return self

    def predict(self, X: csr_matrix) -> np.ndarray:
        return np.argmax(self._scores(X), axis=1)



# 6.  EVALUATION METRICS, written from scratch


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """M[t, p] = number of samples with true label t predicted as p."""
    flat = np.asarray(y_true) * n_classes + np.asarray(y_pred)
    return np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def precision_recall_f1(y_true, y_pred, n_classes):
    """Per-class precision, recall, F1 and support, derived from M."""
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    true_positives = np.diag(matrix).astype(float)
    predicted_positives = matrix.sum(axis=0).astype(float)   # TP + FP
    actual_positives = matrix.sum(axis=1).astype(float)      # TP + FN

    def divide(numerator, denominator):          # 0/0 is defined as 0
        result = np.zeros_like(numerator, dtype=float)
        nonzero = denominator > 0
        result[nonzero] = numerator[nonzero] / denominator[nonzero]
        return result

    precision = divide(true_positives, predicted_positives)
    recall = divide(true_positives, actual_positives)
    f1 = divide(2 * precision * recall, precision + recall)
    return precision, recall, f1, actual_positives


def macro_f1(y_true, y_pred, n_classes) -> float:
    """Unweighted mean of the per-class F1 scores."""
    _, _, f1, _ = precision_recall_f1(y_true, y_pred, n_classes)
    return float(f1.mean())


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def classification_report(y_true, y_pred, labels) -> str:
    precision, recall, f1, support = precision_recall_f1(y_true, y_pred, len(labels))
    lines = [f"{'label':<12}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>9}",
             "-" * 50]
    for i in np.argsort(-f1):
        lines.append(f"{labels[i]:<12}{precision[i]:>11.4f}{recall[i]:>9.4f}"
                     f"{f1[i]:>9.4f}{int(support[i]):>9d}")
    lines.append("-" * 50)
    lines.append(f"{'macro avg':<12}{precision.mean():>11.4f}{recall.mean():>9.4f}"
                 f"{f1.mean():>9.4f}{int(support.sum()):>9d}")
    lines.append(f"{'accuracy':<12}{'':>11}{'':>9}"
                 f"{accuracy(y_true, y_pred):>9.4f}{int(support.sum()):>9d}")
    return "\n".join(lines)


def top_confusions(y_true, y_pred, labels, k=8):
    matrix = confusion_matrix(y_true, y_pred, len(labels))
    np.fill_diagonal(matrix, 0)
    pairs = []
    for flat_index in np.argsort(matrix, axis=None)[::-1][:k]:
        true_index, predicted_index = divmod(int(flat_index), len(labels))
        if matrix[true_index, predicted_index] == 0:
            break
        pairs.append((labels[true_index], labels[predicted_index],
                      int(matrix[true_index, predicted_index])))
    return pairs



# 7.  DATA LOADING


def load_split(path: Path):
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return ([row["sentence"] for row in rows],
            [row["label"] for row in rows],
            [row["language"] for row in rows])



# 8.  MAIN



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "Language_Identification_IndicCorpV2" / "outputs" / "data",
                        help="folder holding train.csv / validation.csv / test.csv "
                             "from Assignment 1")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--l2", type=float, default=1e-6)
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # ---- load the Assignment 1 dataset --------------------------------
    print("=" * 70)
    print("STEP 1  Loading the dataset built in Assignment 1")
    print("=" * 70)
    train_text, train_label, _ = load_split(arguments.data_dir / "train.csv")
    val_text, val_label, _ = load_split(arguments.data_dir / "validation.csv")
    test_text, test_label, test_language = load_split(arguments.data_dir / "test.csv")

    labels = sorted(set(train_label))
    label_to_index = {label: index for index, label in enumerate(labels)}
    language_of = dict(zip(test_label, test_language))

    y_train = np.array([label_to_index[l] for l in train_label])
    y_val = np.array([label_to_index[l] for l in val_label])
    y_test = np.array([label_to_index[l] for l in test_label])

    print(f"    train / validation / test : {len(train_text):,} / {len(val_text):,} / {len(test_text):,}")
    print(f"    languages                 : {len(labels)}")

    # ---- tokenise once, reuse for all six combinations -----------------
    print()
    print("=" * 70)
    print("STEP 2  Tokenising and building the RAW count matrices")
    print("=" * 70)
    train_terms = [extract_features(s) for s in train_text]
    val_terms = [extract_features(s) for s in val_text]
    test_terms = [extract_features(s) for s in test_text]

    vocabulary, df_array = build_vocabulary(train_terms, arguments.min_df)

    counts_train = build_count_matrix(train_terms, vocabulary)
    counts_val = build_count_matrix(val_terms, vocabulary)
    counts_test = build_count_matrix(test_terms, vocabulary)
    print(f"    non-zeros in train   : {counts_train.nnz:,}")

    # ---- run the six combinations --------------------------------------
    print()
    print("=" * 70)
    print("STEP 3  Training the six TF x IDF combinations")
    print("=" * 70)

    results = []
    best = None
    for number, (tf_mode, idf_mode) in enumerate(COMBINATIONS, start=1):
        name = f"TF={tf_mode}, IDF={idf_mode}"
        print(f"\n  [{number}/6] {name}")
        combination_started = time.time()

        tf_train = normalize_tf(counts_train, tf_mode)
        tf_val = normalize_tf(counts_val, tf_mode)
        tf_test = normalize_tf(counts_test, tf_mode)

        idf = compute_idf(df_array, counts_train.shape[0], idf_mode)

        X_train = apply_idf(tf_train, idf)
        X_val = apply_idf(tf_val, idf)
        X_test = apply_idf(tf_test, idf)

        model = SoftmaxRegression(len(vocabulary), len(labels),
                                  learning_rate=arguments.learning_rate,
                                  l2=arguments.l2, epochs=arguments.epochs,
                                  batch_size=arguments.batch_size).fit(
            X_train, y_train, X_val, y_val)

        test_prediction = model.predict(X_test)
        record = {
            "combination": number,
            "tf": tf_mode,
            "idf": idf_mode,
            "tf_normalized": tf_mode != "raw",
            "idf_normalized": idf_mode != "raw",
            "mean_feature_value": float(X_train.data.mean()),
            "val_macro_f1": model.best_val_f1,
            "val_accuracy": accuracy(y_val, model.predict(X_val)),
            "test_macro_f1": macro_f1(y_test, test_prediction, len(labels)),
            "test_accuracy": accuracy(y_test, test_prediction),
            "seconds": time.time() - combination_started,
            "history": model.history,
        }
        results.append(record)
        print(f"      -> TEST macro-F1={record['test_macro_f1']:.4f}  "
              f"accuracy={record['test_accuracy']:.4f}  ({record['seconds']:.0f}s)")

        if best is None or record["test_macro_f1"] > best[0]["test_macro_f1"]:
            best = (record, test_prediction)

    # ---- report ---------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 4  Writing the results file")
    print("=" * 70)
    write_results(arguments, labels, language_of, results, best,
                  y_test, test_text, len(vocabulary), counts_train.nnz,
                  time.time() - started)

    print(f"\n    results -> {arguments.output_dir / 'RESULTS.md'}")
    ranking = sorted(results, key=lambda r: -r["test_macro_f1"])
    print(f"\n    BEST: TF={ranking[0]['tf']}, IDF={ranking[0]['idf']}  "
          f"test macro-F1={ranking[0]['test_macro_f1']:.4f}")
    print(f"    WORST: TF={ranking[-1]['tf']}, IDF={ranking[-1]['idf']}  "
          f"test macro-F1={ranking[-1]['test_macro_f1']:.4f}")


def write_results(arguments, labels, language_of, results, best,
                  y_test, test_text, vocabulary_size, nnz, elapsed) -> None:
    """Write everything into one consolidated Markdown file (+ a JSON copy)."""
    best_record, best_prediction = best
    ranking = sorted(results, key=lambda r: -r["test_macro_f1"])

    lines = [
        "# Assignment 2 - TF and IDF Normalization",
        "",
        "Language identification on the 23-language IndicCorpV2 dataset built in",
        "Assignment 1, comparing normalized and unnormalized TF and IDF vectors.",
        "",
        f"*Generated in {elapsed / 60:.1f} minutes. No sklearn was used.*",
        "",
        "## 1. Setup",
        "",
        "| item | value |",
        "|:--|--:|",
        f"| languages | {len(labels)} |",
        "| sentences per language | 1000 |",
        "| train / validation / test | 18,400 / 2,300 / 2,300 |",
        f"| features (word 1-2 grams + char 2-4 grams, min_df={arguments.min_df}) | {vocabulary_size:,} |",
        f"| non-zeros in the training matrix | {nnz:,} |",
        f"| epochs / learning rate / batch | {arguments.epochs} / {arguments.learning_rate} / {arguments.batch_size} |",
        "",
        "## 2. The formulas being compared",
        "",
        "For a term `t` in sentence `d`, let `f` be its raw count, `N` the number of",
        "training sentences, and `df(t)` the number of training sentences containing `t`.",
        "",
        "**TF variants**",
        "",
        "| variant | formula | normalized? |",
        "|:--|:--|:--|",
        "| raw | `tf = f` | no (Assignment 1) |",
        "| length | `tf = f / (total terms in d)` | yes - task 1(a) |",
        "| maxfreq | `tf = f / max_t' f(t', d)` | yes - task 1(b) |",
        "",
        "**IDF variants**",
        "",
        "| variant | formula | normalized? |",
        "|:--|:--|:--|",
        "| raw | `idf = log(N / df(t))` | no |",
        "| norm | `idf = log(df_max / df(t))` | yes - task 2 |",
        "",
        "`df_max` is the document frequency of the most frequent term across all",
        "sentences, so the normalized IDF is 0 for that term and grows for rarer ones.",
        "",
        "> **Why L2 row normalization is switched off.** Both TF normalizations divide a",
        "> whole row by one per-sentence constant. L2 normalization also divides a row by",
        "> a per-row constant, so it would cancel them exactly and all three TF variants",
        "> would produce identical matrices. The comparison only means something with L2 off.",
        "",
        "## 3. Results - all 6 combinations",
        "",
        "| # | TF | IDF | TF norm. | IDF norm. | mean feature value | val macro-F1 | **test macro-F1** | test accuracy |",
        "|--:|:--|:--|:-:|:-:|--:|--:|--:|--:|",
    ]
    for record in results:
        lines.append(
            f"| {record['combination']} | {record['tf']} | {record['idf']} | "
            f"{'yes' if record['tf_normalized'] else 'no'} | "
            f"{'yes' if record['idf_normalized'] else 'no'} | "
            f"{record['mean_feature_value']:.4g} | {record['val_macro_f1']:.4f} | "
            f"**{record['test_macro_f1']:.4f}** | {record['test_accuracy']:.4f} |")

    lines += [
        "",
        "Ranked by test macro-F1:",
        "",
        "| rank | combination | test macro-F1 |",
        "|--:|:--|--:|",
    ]
    for rank, record in enumerate(ranking, start=1):
        lines.append(f"| {rank} | TF={record['tf']}, IDF={record['idf']} | {record['test_macro_f1']:.4f} |")

    spread = ranking[0]["test_macro_f1"] - ranking[-1]["test_macro_f1"]

    # marginal effect of each normalization, averaged over the other axis
    def average(predicate):
        chosen = [r["test_macro_f1"] for r in results if predicate(r)]
        return sum(chosen) / len(chosen)

    lines += [
        "",
        "## 4. Marginal effect of each normalization",
        "",
        "Average test macro-F1 over the other axis:",
        "",
        "| setting | average test macro-F1 |",
        "|:--|--:|",
        f"| TF raw (unnormalized) | {average(lambda r: r['tf'] == 'raw'):.4f} |",
        f"| TF length-normalized | {average(lambda r: r['tf'] == 'length'):.4f} |",
        f"| TF maxfreq-normalized | {average(lambda r: r['tf'] == 'maxfreq'):.4f} |",
        "| | |",
        f"| IDF unnormalized | {average(lambda r: r['idf'] == 'raw'):.4f} |",
        f"| IDF normalized | {average(lambda r: r['idf'] == 'norm'):.4f} |",
        "",
        f"Spread between the best and worst combination: **{spread:.4f}** macro-F1.",
        "",
        "## 5. Best combination in detail",
        "",
        f"**TF={best_record['tf']}, IDF={best_record['idf']}** - "
        f"test macro-F1 {best_record['test_macro_f1']:.4f}, "
        f"accuracy {best_record['test_accuracy']:.4f}.",
        "",
        "### Per-language scores",
        "",
        "```",
        classification_report(y_test, best_prediction, labels),
        "```",
        "",
        "### Most frequent confusions",
        "",
        "| true | predicted | count |",
        "|:--|:--|--:|",
    ]
    for true_label, predicted_label, count in top_confusions(y_test, best_prediction, labels):
        lines.append(f"| {true_label} ({language_of.get(true_label, '')}) | "
                     f"{predicted_label} ({language_of.get(predicted_label, '')}) | {count} |")

    lines += [
        "",
        "## 6. Training curves (validation macro-F1 per epoch)",
        "",
        "| epoch | " + " | ".join(f"TF={r['tf']}/IDF={r['idf']}" for r in results) + " |",
        "|--:|" + "--:|" * len(results),
    ]
    for epoch_index in range(arguments.epochs):
        row = [str(epoch_index + 1)]
        row += [f"{r['history'][epoch_index]['val_macro_f1']:.4f}" for r in results]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## 7. Sample predictions from the best model",
        "",
        "| sentence (first 60 chars) | true | predicted | correct |",
        "|:--|:--|:--|:-:|",
    ]
    rng = np.random.default_rng(0)
    for row_index in rng.choice(len(y_test), size=12, replace=False):
        sentence = test_text[row_index][:60].replace("|", " ")
        true_label = labels[y_test[row_index]]
        predicted_label = labels[best_prediction[row_index]]
        mark = "yes" if true_label == predicted_label else "NO"
        lines.append(f"| {sentence} | {true_label} | {predicted_label} | {mark} |")

    lines += ["", "## 8. How to reproduce", "", "```", "python assignment2.py", "```", ""]

    (arguments.output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    json_payload = {
        "labels": labels,
        "vocabulary_size": vocabulary_size,
        "settings": {"min_df": arguments.min_df, "epochs": arguments.epochs,
                     "learning_rate": arguments.learning_rate,
                     "batch_size": arguments.batch_size, "l2": arguments.l2},
        "results": results,
    }
    (arguments.output_dir / "results.json").write_text(
        json.dumps(json_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
