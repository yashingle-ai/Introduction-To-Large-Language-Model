"""
Assignment 3 -- BPE and WordPiece tokenization on IndicCorp v2 (Hindi)
=====================================================================

What this script does, in order:

  STEP 1  Download Hindi text from IndicCorp v2 and split the paragraphs
          into sentences.                                        -> data/sentences.txt
  STEP 2  Build train / dev / test splits.  Dev and test hold 1000
          sentences each and are drawn from 10 fixed sentence-length
          bins (100 per bin), so they deliberately contain sentences of
          very different lengths.                                -> data/{train,dev,test}.txt
  STEP 3  Train a BPE and a WordPiece tokenizer for every combination of
              4 training sizes  x  3 vocabulary sizes  x  2 algorithms = 24 runs
          and tokenize the dev and test sets with each of them.
  STEP 4  Write the report.                          -> outputs/RESULTS.md, results.json

The corpus stores one paragraph per line, so STEP 1 must sentence-tokenize.
The file for Hindi (data/hi-1.txt) is 26 GB, so we never download it whole:
we ask for a handful of byte ranges spread across the file and read only
those.  Ranges are spread out (not taken from the head) because the head of
these crawl dumps is usually one single news domain.

Run:
    python assignment3.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

HF_URL = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/hi-1.txt"

# --- the four experiment axes from the assignment --------------------------
TRAIN_SIZES = [100_000, 300_000, 500_000, 1_000_000]   # task 3
VOCAB_SIZES = [20_000, 30_000, 50_000]                 # task 4
ALGORITHMS = ["BPE", "WordPiece"]                      # task 2

DEV_SIZE = TEST_SIZE = 1000                            # task 1

# Ten sentence-length bins (in words).  Dev and test take 100 sentences from
# each bin, which is the "maintain a length distribution" hint of task 1: the
# two sets are flat across lengths instead of being all medium-sized sentences.
LENGTH_BINS = [(3, 5), (6, 8), (9, 11), (12, 15), (16, 19),
               (20, 24), (25, 29), (30, 39), (40, 59), (60, 10_000)]
PER_BIN = 100                                          # 10 bins x 100 = 1000



# STEP 1.  DOWNLOAD AND SENTENCE-TOKENIZE


# A sentence ends at a danda, a double danda, '?', '!' or '.', followed by
# space.  The lookbehind keeps the punctuation attached to the sentence.
SENTENCE_END = re.compile(r"(?<=[।॥?!.])\s+")


def sentence_tokenize(paragraph: str) -> list[str]:
    """Split one corpus row (a paragraph) into sentences."""
    return [s.strip() for s in SENTENCE_END.split(paragraph) if s.strip()]


def is_devanagari(char: str) -> bool:
    return "ऀ" <= char <= "ॿ"


def clean(sentence: str) -> str:
    """NFC-normalize and collapse every run of whitespace to a single space.

    `split()` treats the exotic Unicode line boundaries (U+2028, U+0085, form
    feed, ...) as whitespace, so this also guarantees no sentence can contain
    something that `splitlines()` would later break in half - without it, one
    sentence written to the cache file comes back as two.
    """
    return " ".join(unicodedata.normalize("NFC", sentence).split())


def usable(sentence: str) -> bool:
    """Keep normal running text; drop menus, tables, ad slugs and stubs."""
    if not 20 <= len(sentence) <= 600:
        return False
    if len(sentence.split()) < 3:
        return False
    letters = [c for c in sentence if c.isalpha()]
    if not letters:
        return False
    # at least 80% of the letters must be Devanagari, otherwise it is an
    # English/transliterated line that happened to sit in the Hindi file
    return sum(is_devanagari(c) for c in letters) / len(letters) >= 0.80


def fetch_window(url: str, start: int, length: int, retries: int = 4) -> str:
    """Download bytes [start, start+length) and decode as UTF-8.

    The window starts and ends mid-line, so the caller throws away the first
    and the last line; `errors="ignore"` drops the two broken characters at
    the seams.
    """
    headers = {"Range": f"bytes={start}-{start + length - 1}"}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=300)
            response.raise_for_status()
            return response.content.decode("utf-8", errors="ignore")
        except Exception:                       # the HF CDN is occasionally flaky
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    return ""


def download_sentences(cache: Path, target: int, window_mb: int, workers: int) -> list[str]:
    """Return `target` unique, clean Hindi sentences (cached on disk)."""
    if cache.exists():
        # split("\n"), not splitlines(), so the count matches what was written
        sentences = [clean(s) for s in cache.read_text(encoding="utf-8").split("\n")]
        if len(sentences) >= target:
            print(f"    cached: {len(sentences):,} sentences in {cache.name}")
            return sentences[:target]

    size = int(requests.head(HF_URL, allow_redirects=True, timeout=60)
               .headers["Content-Length"])
    window = window_mb * 1_000_000
    print(f"    corpus file: {size / 1e9:.1f} GB, reading {window_mb} MB windows")

    # 128 candidate offsets evenly spread over the whole file, then shuffled,
    # so the windows we actually use are scattered instead of consecutive.
    offsets = [int((i + 0.5) / 128 * (size - window)) for i in range(128)]
    random.Random(0).shuffle(offsets)

    sentences: list[str] = []
    seen: set[str] = set()
    used = 0
    started = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch_start in range(0, len(offsets), workers):
            if len(sentences) >= target:
                break
            batch = offsets[batch_start:batch_start + workers]
            for text in pool.map(lambda o: fetch_window(HF_URL, o, window), batch):
                used += 1
                for paragraph in text.split("\n")[1:-1]:     # drop partial lines
                    for sentence in sentence_tokenize(paragraph.strip()):
                        sentence = clean(sentence)
                        if not usable(sentence) or sentence in seen:
                            continue
                        seen.add(sentence)
                        sentences.append(sentence)
            print(f"    {used:3d} windows ({used * window_mb} MB)  ->  "
                  f"{len(sentences):,} sentences   [{time.time() - started:.0f}s]",
                  flush=True)

    if len(sentences) < target:
        raise RuntimeError(f"only {len(sentences):,} sentences, need {target:,}")

    sentences = sentences[:target]
    cache.write_text("\n".join(sentences), encoding="utf-8")
    return sentences



# STEP 2.  TRAIN / DEV / TEST SPLIT, STRATIFIED BY SENTENCE LENGTH


def bin_of(sentence: str) -> int:
    """Index of the length bin this sentence falls into (-1 if none)."""
    n_words = len(sentence.split())
    for index, (low, high) in enumerate(LENGTH_BINS):
        if low <= n_words <= high:
            return index
    return -1


def make_splits(sentences: list[str], seed: int):
    """Dev and test get 100 sentences from each of the 10 length bins."""
    rng = random.Random(seed)

    buckets: list[list[str]] = [[] for _ in LENGTH_BINS]
    for sentence in sentences:
        index = bin_of(sentence)
        if index >= 0:
            buckets[index].append(sentence)

    dev: list[str] = []
    test: list[str] = []
    held_out: set[str] = set()
    bin_report = []

    for index, bucket in enumerate(buckets):
        rng.shuffle(bucket)
        need = 2 * PER_BIN
        if len(bucket) < need:
            raise RuntimeError(f"bin {LENGTH_BINS[index]} has only {len(bucket)} sentences")
        dev.extend(bucket[:PER_BIN])
        test.extend(bucket[PER_BIN:need])
        held_out.update(bucket[:need])
        bin_report.append({"bin": f"{LENGTH_BINS[index][0]}-{LENGTH_BINS[index][1]} words",
                           "pool": len(bucket), "dev": PER_BIN, "test": PER_BIN})

    rng.shuffle(dev)
    rng.shuffle(test)

    # everything not held out is training data
    train = [s for s in sentences if s not in held_out]
    rng.shuffle(train)
    return train, dev, test, bin_report


def length_stats(sentences: list[str]) -> dict:
    words = [len(s.split()) for s in sentences]
    chars = [len(s) for s in sentences]
    words_sorted = sorted(words)
    return {
        "sentences": len(sentences),
        "mean_words": sum(words) / len(words),
        "min_words": words_sorted[0],
        "median_words": words_sorted[len(words) // 2],
        "max_words": words_sorted[-1],
        "mean_chars": sum(chars) / len(chars),
    }



# STEP 3.  TRAIN THE TOKENIZERS AND TOKENIZE DEV / TEST


def build_tokenizer(algorithm: str, corpus: list[str], vocab_size: int) -> Tokenizer:
    """Train one BPE or one WordPiece tokenizer on `corpus`.

    Both use the same whitespace/punctuation pre-tokenizer so the only thing
    that differs between them is the subword algorithm itself:
      * BPE       starts from characters and repeatedly merges the most
                  frequent adjacent pair.
      * WordPiece starts from characters and adds the pair that most
                  increases the likelihood of the data, i.e. it scores a pair
                  by  count(ab) / (count(a) * count(b))  instead of count(ab).
    """
    if algorithm == "BPE":
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", continuing_subword_prefix="##"))
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=2, special_tokens=["[UNK]"],
            continuing_subword_prefix="##", show_progress=False)
    else:
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=200))
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size, min_frequency=2, special_tokens=["[UNK]"],
            continuing_subword_prefix="##", show_progress=False)

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(corpus, trainer)
    return tokenizer


def tokenize_split(tokenizer: Tokenizer, sentences: list[str]) -> tuple[list[list[str]], dict]:
    """Tokenize a split and measure how it was tokenized."""
    encodings = tokenizer.encode_batch(sentences)

    all_tokens = [encoding.tokens for encoding in encodings]
    n_tokens = sum(len(t) for t in all_tokens)
    n_words = sum(len(set(encoding.word_ids)) for encoding in encodings)

    unk = sum(t.count("[UNK]") for t in all_tokens)
    continuations = sum(sum(tok.startswith("##") for tok in t) for t in all_tokens)
    char_total = sum(len(tok.lstrip("#")) for t in all_tokens for tok in t)

    # Walk the words of every sentence once and collect, per word:
    #   * was it kept whole (exactly one token)?
    #   * is it a real word rather than a piece of punctuation?
    # The pre-tokenizer splits punctuation off as its own "word", and a comma
    # or a danda is always exactly one token, so the plain fertility number is
    # pulled towards 1 by punctuation.  The *_alpha_* figures exclude it and
    # are the honest measure of how hard the tokenizer works on real words.
    whole_words = 0
    alpha_words = 0
    alpha_tokens = 0
    alpha_whole = 0
    for encoding in encodings:
        groups: dict[int, list[str]] = {}
        for token, word_id in zip(encoding.tokens, encoding.word_ids):
            groups.setdefault(word_id, []).append(token)
        for tokens in groups.values():
            if len(tokens) == 1:
                whole_words += 1
            if any(character.isalpha() for token in tokens for character in token):
                alpha_words += 1
                alpha_tokens += len(tokens)
                if len(tokens) == 1:
                    alpha_whole += 1

    used_types = {tok for t in all_tokens for tok in t}

    return all_tokens, {
        "sentences": len(sentences),
        "words": n_words,
        "tokens": n_tokens,
        "tokens_per_sentence": n_tokens / len(sentences),
        "tokens_per_word": n_tokens / n_words,               # "fertility"
        "alpha_words": alpha_words,
        "tokens_per_alpha_word": alpha_tokens / alpha_words,  # punctuation excluded
        "chars_per_token": char_total / n_tokens,
        "pct_words_kept_whole": 100 * whole_words / n_words,
        "pct_alpha_words_kept_whole": 100 * alpha_whole / alpha_words,
        "pct_continuation_tokens": 100 * continuations / n_tokens,
        "unk_tokens": unk,
        "pct_unk": 100 * unk / n_tokens,
        "distinct_tokens_used": len(used_types),
    }


def run_experiments(train, dev, test, out_dir: Path, results_path: Path,
                    report_fn) -> list[dict]:
    """The 4 x 3 x 2 grid.  Results are flushed after every run."""
    token_dir = out_dir / "tokenized"
    model_dir = out_dir / "tokenizers"
    token_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total = len(TRAIN_SIZES) * len(VOCAB_SIZES) * len(ALGORITHMS)
    number = 0

    # smallest training sets first, so partial results are still a complete
    # picture of the cheap end of the grid if the run is cut short
    for train_size in TRAIN_SIZES:
        corpus = train[:train_size]
        for vocab_size in VOCAB_SIZES:
            for algorithm in ALGORITHMS:
                number += 1
                tag = f"{algorithm.lower()}_train{train_size}_vocab{vocab_size}"
                print(f"\n  [{number:2d}/{total}] {algorithm:<9} "
                      f"train={train_size:,}  vocab={vocab_size:,}", flush=True)

                started = time.time()
                tokenizer = build_tokenizer(algorithm, corpus, vocab_size)
                train_seconds = time.time() - started

                dev_tokens, dev_stats = tokenize_split(tokenizer, dev)
                test_tokens, test_stats = tokenize_split(tokenizer, test)

                for name, tokens in (("dev", dev_tokens), ("test", test_tokens)):
                    (token_dir / f"{tag}.{name}.txt").write_text(
                        "\n".join(" ".join(t) for t in tokens), encoding="utf-8")
                tokenizer.save(str(model_dir / f"{tag}.json"))

                vocab = tokenizer.get_vocab()
                record = {
                    "run": number,
                    "algorithm": algorithm,
                    "train_size": train_size,
                    "vocab_size_requested": vocab_size,
                    "vocab_size_actual": len(vocab),
                    "single_char_types": sum(1 for t in vocab if len(t.lstrip("#")) == 1),
                    "mean_type_length": sum(len(t.lstrip("#")) for t in vocab) / len(vocab),
                    "train_seconds": train_seconds,
                    "dev": dev_stats,
                    "test": test_stats,
                    "example": " ".join(tokenizer.encode(EXAMPLE).tokens),
                }
                results.append(record)

                print(f"      vocab built = {record['vocab_size_actual']:,}   "
                      f"dev tokens/word = {dev_stats['tokens_per_word']:.3f}   "
                      f"test tokens/word = {test_stats['tokens_per_word']:.3f}   "
                      f"UNK = {test_stats['pct_unk']:.3f}%   ({train_seconds:.0f}s)",
                      flush=True)

                results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
                report_fn(results)                    # rewrite RESULTS.md each time

    return results


EXAMPLE = "भारतीय अंतरिक्ष अनुसंधान संगठन ने चंद्रयान अभियान की सफलता की घोषणा की।"



# STEP 4.  COMPARING THE TOKENIZERS AGAINST EACH OTHER


def tag_of(record: dict) -> str:
    return (f"{record['algorithm'].lower()}_train{record['train_size']}"
            f"_vocab{record['vocab_size_requested']}")


def load_vocab(out_dir: Path, tag: str) -> set[str]:
    blob = json.loads((out_dir / "tokenizers" / f"{tag}.json").read_text(encoding="utf-8"))
    return set(blob["model"]["vocab"])


def load_tokenized(out_dir: Path, tag: str, split: str) -> list[str]:
    return (out_dir / "tokenized" / f"{tag}.{split}.txt").read_text(
        encoding="utf-8").splitlines()


def jaccard(left: set[str], right: set[str]) -> float:
    return 100 * len(left & right) / len(left | right)


def agreement(left: list[str], right: list[str]) -> float:
    """% of sentences that come out with exactly the same token sequence."""
    return 100 * sum(a == b for a, b in zip(left, right)) / len(left)


def compare(results: list[dict], out_dir: Path, test: list[str]) -> dict:
    """Measure how much two tokenizers really differ.

    Fertility can saturate - once a vocabulary covers the frequent words, more
    training data barely changes the average tokens-per-word.  That does NOT
    mean the tokenizers are the same, so we also compare the vocabularies
    themselves and count how many test sentences come out differently.
    
    """
    sizes = sorted({r["train_size"] for r in results})
    vocabs = sorted({r["vocab_size_requested"] for r in results})
    largest, middle_vocab = sizes[-1], (30_000 if 30_000 in vocabs else vocabs[0])
    by_key = {(r["algorithm"], r["train_size"], r["vocab_size_requested"]): r
              for r in results}

    analysis: dict = {"size_effect": [], "vocab_effect": [], "algorithm_effect": [],
                      "examples": []}

    # (a) each training size against the largest one, vocabulary held fixed
    for algorithm in ALGORITHMS:
        reference = by_key.get((algorithm, largest, middle_vocab))
        if reference is None:
            continue
        ref_vocab = load_vocab(out_dir, tag_of(reference))
        ref_tokens = load_tokenized(out_dir, tag_of(reference), "test")
        for size in sizes:
            record = by_key.get((algorithm, size, middle_vocab))
            if record is None:
                continue
            analysis["size_effect"].append({
                "algorithm": algorithm, "train_size": size, "vocab_size": middle_vocab,
                "vocab_overlap_with_largest": jaccard(load_vocab(out_dir, tag_of(record)),
                                                      ref_vocab),
                "test_agreement_with_largest": agreement(
                    load_tokenized(out_dir, tag_of(record), "test"), ref_tokens),
            })

    # (b) each vocabulary against the largest one, training size held fixed
    for algorithm in ALGORITHMS:
        reference = by_key.get((algorithm, largest, vocabs[-1]))
        if reference is None:
            continue
        ref_tokens = load_tokenized(out_dir, tag_of(reference), "test")
        for vocab in vocabs:
            record = by_key.get((algorithm, largest, vocab))
            if record is None:
                continue
            analysis["vocab_effect"].append({
                "algorithm": algorithm, "vocab_size": vocab,
                "test_agreement_with_largest_vocab": agreement(
                    load_tokenized(out_dir, tag_of(record), "test"), ref_tokens),
            })

    # (c) BPE against WordPiece at the same setting
    for vocab in vocabs:
        bpe = by_key.get(("BPE", largest, vocab))
        wordpiece = by_key.get(("WordPiece", largest, vocab))
        if not (bpe and wordpiece):
            continue
        analysis["algorithm_effect"].append({
            "train_size": largest, "vocab_size": vocab,
            "vocab_overlap": jaccard(load_vocab(out_dir, tag_of(bpe)),
                                     load_vocab(out_dir, tag_of(wordpiece))),
            "test_agreement": agreement(load_tokenized(out_dir, tag_of(bpe), "test"),
                                        load_tokenized(out_dir, tag_of(wordpiece), "test")),
        })

    # (d) real test sentences where the settings visibly disagree.  A fixed
    # hand-picked sentence is usually made of frequent words and comes out
    # identical everywhere, which demonstrates nothing - so pick the sentences
    # that actually changed, ranked by how much they changed.
    small = by_key.get(("BPE", sizes[0], vocabs[0]))
    big = by_key.get(("BPE", largest, vocabs[-1]))
    if small and big:
        small_tokens = load_tokenized(out_dir, tag_of(small), "test")
        big_tokens = load_tokenized(out_dir, tag_of(big), "test")
        differences = [
            (len(a.split()) - len(b.split()), index)
            for index, (a, b) in enumerate(zip(small_tokens, big_tokens)) if a != b
        ]
        differences.sort(reverse=True)
        for _, index in differences[:4]:
            analysis["examples"].append({
                "sentence": test[index],
                "variants": [
                    {"algorithm": r["algorithm"], "train_size": r["train_size"],
                     "vocab_size": r["vocab_size_requested"],
                     "tokens": load_tokenized(out_dir, tag_of(r), "test")[index]}
                    for r in results
                    if r["train_size"] in (sizes[0], largest)
                    and r["vocab_size_requested"] in (vocabs[0], vocabs[-1])
                ],
            })
        analysis["pct_test_sentences_changed"] = 100 * len(differences) / len(small_tokens)

    return analysis



# STEP 5.  REPORT


def pick(results, algorithm=None, train_size=None, vocab_size=None):
    return [r for r in results
            if (algorithm is None or r["algorithm"] == algorithm)
            and (train_size is None or r["train_size"] == train_size)
            and (vocab_size is None or r["vocab_size_requested"] == vocab_size)]




# MAIN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=here / "data")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs")
    parser.add_argument("--window-mb", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    arguments.data_dir.mkdir(parents=True, exist_ok=True)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("=" * 72)
    print("STEP 1  Downloading Hindi text and splitting paragraphs into sentences")
    print("=" * 72)
    needed = max(TRAIN_SIZES) + DEV_SIZE + TEST_SIZE + 20_000   # margin for binning
    sentences = download_sentences(arguments.data_dir / "sentences.txt",
                                   needed, arguments.window_mb, arguments.workers)
    corpus_stats = length_stats(sentences)
    print(f"    {corpus_stats['sentences']:,} sentences, "
          f"{corpus_stats['mean_words']:.1f} words / {corpus_stats['mean_chars']:.0f} chars each")

    print()
    print("=" * 72)
    print("STEP 2  Building length-stratified train / dev / test splits")
    print("=" * 72)
    train, dev, test, bin_report = make_splits(sentences, arguments.seed)
    for name, split in (("train", train), ("dev", dev), ("test", test)):
        (arguments.data_dir / f"{name}.txt").write_text("\n".join(split), encoding="utf-8")
    split_stats = {"train": length_stats(train), "dev": length_stats(dev),
                   "test": length_stats(test)}
    for name in ("train", "dev", "test"):
        s = split_stats[name]
        print(f"    {name:<5} {s['sentences']:>9,}  words min/median/max = "
              f"{s['min_words']}/{s['median_words']}/{s['max_words']}  "
              f"mean {s['mean_words']:.1f}")

    if len(train) < max(TRAIN_SIZES):
        raise RuntimeError(f"training split has {len(train):,} sentences, "
                           f"need {max(TRAIN_SIZES):,}")

    print()
    print("=" * 72)
    print("STEP 3  Training 4 sizes x 3 vocabularies x 2 algorithms = 24 tokenizers")
    print("=" * 72)

    # def report(current):
    #     write_report(current, corpus_stats, bin_report, split_stats, test,
    #                  arguments.output_dir, time.time() - started)

    results = run_experiments(train, dev, test, arguments.output_dir,
                              arguments.output_dir / "results.json", report)

    print()
    print("=" * 72)
    print(f"DONE  {len(results)} configurations in {(time.time() - started) / 60:.1f} minutes")
    print(f"      report -> {arguments.output_dir / 'RESULTS.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
