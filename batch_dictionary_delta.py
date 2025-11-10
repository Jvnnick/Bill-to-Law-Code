#!/usr/bin/env python3
"""
batch_dictionary_delta.py

Walk a root directory containing state / legislative period folders. For each law folder
that ships a `Rohtexte/GE/*.txt` (bill) and `Rohtexte/LR/*.txt` (law) file, compute
dictionary match totals and their deltas using the preprocessing pipeline defined in
`Notebooks/Dictionary_Approach_V4.ipynb`.

Key features
------------
* Reads TXT files with UTF-8 + Latin-1 fallback (skip pair on persistent errors).
* Applies notebook-identical preprocessing: clean LaTeX artefacts, fix line-break
  hyphens, lowercase, strip URLs & symbols (digits + hyphens preserved), tokenize with
  NLTK, optional stopword removal, Snowball stemming (German), join into
  `text_processed`.
* Builds unigram DTMs via `CountVectorizer(token_pattern=r"(?u)\\b\\w+(?:-\\w+)*\\b")`.
* Aggregates per-category dictionary counts (dictionary stems produced with the same
  stemmer) and total matches per document.
* Writes per-pair outputs (`doc_summary_pair.csv`, `delta_summary.csv`) into a
  `Results/` subfolder beside `Rohtexte`.
* Appends / updates global `pair_deltas.csv` and `skipped.csv` logs at the chosen root.
* Deterministic ordering, dictionary hash logging, idempotent row sums, configurable CLI.

Usage example
-------------
    python batch_dictionary_delta.py \\
        --root /path/to/root \\
        --dict Other/dictionary_li2025.json \\
        --out-global /path/to/root/pair_deltas.csv \\
        --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

try:
    from tqdm import tqdm

    HAS_TQDM = True
except Exception:  # pragma: no cover - tqdm is optional
    HAS_TQDM = False


DOC_ORDER = ("GE", "LR")
TOKEN_PATTERN = r"(?u)\b\w+(?:-\w+)*\b"
WORD_RE = re.compile(r"[a-zäöüß]+[a-z0-9äöüß\-]*|\d+[a-zäöüß][a-z0-9äöüß\-]*", re.IGNORECASE)


@dataclass(frozen=True)
class PairContext:
    pair_dir: Path
    ge_files: Tuple[Path, ...]
    lr_files: Tuple[Path, ...]


def append_local_nltk_path():
    """Include repository-level nltk_data (if present)."""
    try:
        base = Path(__file__).resolve().parent.parent
    except NameError:  # pragma: no cover - e.g., interactive execution
        base = Path.cwd()
    local = base / "nltk_data"
    if local.is_dir():
        local_str = str(local)
        if local_str not in nltk.data.path:
            nltk.data.path.insert(0, local_str)


append_local_nltk_path()


def extract_state_wp(path_str: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (state, "<NN. WP>") parsed from a directory path."""
    norm = os.path.normpath(path_str)
    parts = norm.split(os.sep)
    wp_re = re.compile(r"^\s*\d{1,2}\.\s*WP\s*$", re.IGNORECASE)
    for idx, segment in enumerate(parts):
        if wp_re.match(segment):
            state = parts[idx - 1] if idx - 1 >= 0 else None
            return state, segment.strip()
    return None, None


def get_state_abbr(state: Optional[str]) -> Optional[str]:
    """Translate federal state names into abbreviations (folding diacritics)."""
    if not state:
        return None

    def _nfc(text: str) -> str:
        import unicodedata

        return unicodedata.normalize("NFC", text.strip())

    def _strip(text: str) -> str:
        import unicodedata

        return "".join(
            ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
        )

    mapping = {
        "Baden-Württemberg": "BW",
        "Baden-Wuerttemberg": "BW",
        "Bayern": "BY",
        "Berlin": "BE",
        "Brandenburg": "BB",
        "Bremen": "HB",
        "Hamburg": "HH",
        "Hessen": "HE",
        "Mecklenburg-Vorpommern": "MV",
        "Niedersachsen": "NI",
        "Nordrhein-Westfalen": "NW",
        "Rheinland-Pfalz": "RP",
        "Saarland": "SL",
        "Sachsen": "SN",
        "Sachsen-Anhalt": "ST",
        "Schleswig-Holstein": "SH",
        "Thüringen": "TH",
        "Thueringen": "TH",
        "Thuringen": "TH",
    }

    candidate = _nfc(state)
    if candidate in mapping:
        return mapping[candidate]

    candidate_ascii = _strip(candidate)
    if candidate_ascii in mapping:
        return mapping[candidate_ascii]

    # Fall back to uppercase alpha characters of the original segment
    letters = re.sub(r"[^A-Z]", "", candidate.upper())
    return letters or candidate


def derive_overunit_id(path: Path) -> Optional[str]:
    """Build OVERUNIT_ID as <STATE_ABBR><WP_DIGITS>."""
    state, wp_segment = extract_state_wp(str(path))
    abbr = get_state_abbr(state)
    if not abbr or not wp_segment:
        return None
    wp_digits = "".join(ch for ch in wp_segment if ch.isdigit())
    return f"{abbr}{wp_digits}" if wp_digits else None


def clean_latex_artifacts(text: str) -> str:
    text = re.sub(r"\$\$|\$", " ", text)
    text = re.sub(r"\\begin\{.*?\}|\\end\{.*?\}", " ", text)
    text = text.replace(r"\$", "§")
    text = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(left|right|int|gathered|text|mathrm)\b", " ", text)
    text = re.sub(r"([a-zA-Z])\(", r"\1 (", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_linebreak_hyphens(text: str) -> str:
    return re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)


def text_lowercase(text: str) -> str:
    return text.lower()


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_symbols(text: str) -> str:
    return re.sub(r"[^0-9A-Za-zÄÖÜäöüß\-\s]", " ", text)


def ensure_tokenizer_available():
    try:
        _ = word_tokenize("probe")
    except LookupError as exc:  # pragma: no cover - hard to trigger when data present
        raise LookupError(
            "NLTK tokenizer data (punkt) not found. "
            "Download via: nltk.download('punkt', download_dir='nltk_data')"
        ) from exc


def load_stopwords() -> set[str]:
    try:
        return set(stopwords.words("german"))
    except LookupError as exc:  # pragma: no cover - same rationale
        raise LookupError(
            "NLTK stopword corpus not found. "
            "Download via: nltk.download('stopwords', download_dir='nltk_data')"
        ) from exc


def preprocess_dataframe(df: pd.DataFrame, remove_stop: bool) -> pd.DataFrame:
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_latex_artifacts)
    df["text_clean"] = df["text_clean"].apply(fix_linebreak_hyphens)
    df["text_clean"] = df["text_clean"].apply(text_lowercase)
    df["text_clean"] = df["text_clean"].apply(remove_urls)
    df["text_clean"] = df["text_clean"].apply(remove_symbols)

    ensure_tokenizer_available()
    df["tokens"] = df["text_clean"].apply(word_tokenize)

    punct = set(string.punctuation)
    df["tokens"] = df["tokens"].apply(lambda toks: [t for t in toks if t not in punct])
    df["tokens"] = df["tokens"].apply(lambda toks: [t for t in toks if len(t) > 1])

    if remove_stop:
        stop_words = load_stopwords()
        df["tokens"] = df["tokens"].apply(lambda toks: [t for t in toks if t.lower() not in stop_words])

    stemmer = SnowballStemmer("german")
    df["tokens"] = df["tokens"].apply(lambda toks: [stemmer.stem(t) for t in toks])
    df["text_processed"] = df["tokens"].apply(lambda toks: " ".join(toks))
    return df


def read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def stem_unigram(term: str, stemmer: SnowballStemmer) -> Optional[str]:
    tokens = WORD_RE.findall(term.lower())
    if not tokens:
        return None
    return stemmer.stem(tokens[0])


def stem_dictionary(raw_dict: Dict[str, Sequence[str]]) -> Tuple[List[str], Dict[str, set[str]]]:
    stemmer = SnowballStemmer("german")
    categories = list(raw_dict.keys())
    stemmed: Dict[str, set[str]] = {}
    for cat in categories:
        stems = set()
        for term in raw_dict[cat]:
            stem = stem_unigram(term, stemmer)
            if stem:
                stems.add(stem)
        stemmed[cat] = stems
    return categories, stemmed


def hash_dictionary(raw_dict: Dict[str, Sequence[str]]) -> str:
    canon = json.dumps(raw_dict, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


def collect_pair_dirs(root: Path) -> List[PairContext]:
    pair_dirs: Dict[Path, PairContext] = {}
    for roh in sorted(root.rglob("Rohtexte")):
        if not roh.is_dir():
            continue
        ge_dir = roh / "GE"
        lr_dir = roh / "LR"
        if ge_dir.is_dir() and lr_dir.is_dir():
            pair_dir = roh.parent
            ge_txts = tuple(sorted(p for p in ge_dir.glob("*.txt") if p.is_file()))
            lr_txts = tuple(sorted(p for p in lr_dir.glob("*.txt") if p.is_file()))
            pair_dirs[pair_dir] = PairContext(pair_dir=pair_dir, ge_files=ge_txts, lr_files=lr_txts)
    # Deterministic ordering by relative path (fallback to absolute if not possible)
    ordered = sorted(pair_dirs.values(), key=lambda ctx: str(ctx.pair_dir))
    return ordered


def validate_single_txt(files: Sequence[Path]) -> Tuple[bool, Optional[str]]:
    if len(files) == 0:
        return False, "no TXT files found"
    if len(files) > 1:
        return False, f"expected exactly 1 TXT, found {len(files)}"
    return True, None


def build_vectorizer():
    return CountVectorizer(lowercase=False, token_pattern=TOKEN_PATTERN, ngram_range=(1, 1))


def ensure_doc_types(df: pd.DataFrame) -> Optional[str]:
    present = set(df["doc_type"])
    missing = [doc for doc in DOC_ORDER if doc not in present]
    if missing:
        return f"missing required doc types: {', '.join(missing)}"
    return None


def create_results_dir(results_dir: Path, dry_run: bool):
    if dry_run:
        return
    results_dir.mkdir(parents=True, exist_ok=True)


def compute_counts(
    df_processed: pd.DataFrame,
    categories: Sequence[str],
    stemmed_dict: Dict[str, set[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(df_processed["text_processed"])
    vocab = vectorizer.get_feature_names_out()
    dtm_df = pd.DataFrame(X.toarray(), index=df_processed["doc_type"], columns=vocab)

    counts_rows = []
    for doc in DOC_ORDER:
        if doc not in dtm_df.index:
            raise ValueError(f"doc_type '{doc}' not present after vectorization")
        counts = {}
        for cat in categories:
            stems = stemmed_dict[cat]
            relevant = stems.intersection(vocab)
            if relevant:
                cols = sorted(relevant)
                counts[cat] = int(dtm_df.loc[doc, cols].sum())
            else:
                counts[cat] = 0
        counts_rows.append(counts)

    counts_df = pd.DataFrame(counts_rows, index=list(DOC_ORDER))
    counts_df["total_dict_matches"] = counts_df.sum(axis=1)

    delta_data = {}
    for cat in categories:
        delta_data[f"delta_{cat}"] = counts_df.loc["LR", cat] - counts_df.loc["GE", cat]
    delta_data["delta_total_dict_matches"] = (
        counts_df.loc["LR", "total_dict_matches"] - counts_df.loc["GE", "total_dict_matches"]
    )
    delta_df = pd.DataFrame([delta_data])
    return counts_df, delta_df


def relative_path_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_csv(path: Path, df: pd.DataFrame, dry_run: bool):
    if dry_run:
        return
    df.to_csv(path, index=False)


def load_dictionary(dict_path: Path) -> Dict[str, Sequence[str]]:
    with dict_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def should_skip_existing(results_dir: Path, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    expected = [results_dir / "doc_summary_pair.csv", results_dir / "delta_summary.csv"]
    return all(p.exists() for p in expected)


def merge_csv(existing_path: Path, new_df: pd.DataFrame, key_columns: Sequence[str], dry_run: bool):
    if dry_run:
        return
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
    else:
        existing = pd.DataFrame(columns=new_df.columns)
    if key_columns:
        existing = existing[~existing[key_columns[0]].isin(new_df[key_columns[0]])]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(key_columns or list(combined.columns)).reset_index(drop=True)
    combined.to_csv(existing_path, index=False)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dictionary deltas for GE/LR Rohtexte pairs.")
    parser.add_argument("--root", required=True, help="Root directory to traverse for pair folders.")
    parser.add_argument("--dict", required=True, help="Path to dictionary (JSON).")
    parser.add_argument(
        "--out-global",
        help="Path to pair_deltas.csv (defaults to <root>/pair_deltas.csv).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-pair result files.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folders with existing per-pair outputs (default behaviour).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover results but skip writing files.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose progress information.")
    parser.add_argument(
        "--keep-stopwords",
        action="store_true",
        help="Keep stopwords (default removes German stopwords).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar even if tqdm is available.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    dict_path = Path(args.dict).resolve()
    if not dict_path.is_file():
        raise SystemExit(f"Dictionary not found: {dict_path}")

    skip_existing = args.skip_existing or not args.overwrite
    if args.overwrite and args.skip_existing:
        raise SystemExit("Choose either --overwrite or --skip-existing, not both.")

    out_global = Path(args.out_global).resolve() if args.out_global else root / "pair_deltas.csv"
    skipped_log_path = out_global.parent / "skipped.csv"

    raw_dict = load_dictionary(dict_path)
    categories, stemmed_dict = stem_dictionary(raw_dict)
    dictionary_hash = hash_dictionary(raw_dict)
    dictionary_term_count = sum(len(v) for v in raw_dict.values())

    if args.verbose:
        print(f"Dictionary loaded: {dict_path} ({len(categories)} categories, {dictionary_term_count} terms)")
        print(f"Dictionary SHA-256: {dictionary_hash}")

    pair_contexts = collect_pair_dirs(root)
    if not pair_contexts:
        if args.verbose:
            print("No valid pair folders found.")
        return 0

    use_progress = HAS_TQDM and not args.no_progress
    progress_bar = tqdm(pair_contexts, desc="Processing pairs", unit="pair") if use_progress else None
    iterator: Iterable[PairContext] = progress_bar if progress_bar is not None else pair_contexts

    summary_rows = []
    skipped_rows = []

    def log(message: str):
        if not args.verbose:
            return
        if progress_bar is not None:
            progress_bar.write(message)
        else:
            print(message)

    for ctx in iterator:
        pair_dir = ctx.pair_dir
        rel_pair = relative_path_or_str(pair_dir, root)
        results_dir = pair_dir / "Results"

        if should_skip_existing(results_dir, skip_existing):
            log(f"[skip-existing] {rel_pair}")
            continue

        ok_ge, ge_reason = validate_single_txt(ctx.ge_files)
        ok_lr, lr_reason = validate_single_txt(ctx.lr_files)
        if not ok_ge or not ok_lr:
            reason = ge_reason if not ok_ge else lr_reason
            skipped_rows.append({"pair_path": rel_pair, "reason": reason})
            log(f"[skip] {rel_pair} – {reason}")
            continue

        doc_records = []
        doc_paths = {"GE": ctx.ge_files[0], "LR": ctx.lr_files[0]}
        try:
            for doc_type in DOC_ORDER:
                path = doc_paths[doc_type]
                text = read_text_with_fallback(path)
                doc_records.append(
                    {
                        "doc_type": doc_type,
                        "doc_name": path.stem,
                        "doc_path": path,
                        "text": text,
                    }
                )
        except Exception as exc:
            skipped_rows.append({"pair_path": rel_pair, "reason": f"read error: {exc}"})
            log(f"[skip] {rel_pair} – read error: {exc}")
            continue

        df = pd.DataFrame(doc_records)
        df["pair_dir"] = str(pair_dir)
        df["pair_relative_path"] = rel_pair
        df["doc_path"] = df["doc_path"].apply(lambda p: relative_path_or_str(Path(p), root))

        try:
            df_processed = preprocess_dataframe(df, remove_stop=(not args.keep_stopwords))
        except Exception as exc:
            skipped_rows.append({"pair_path": rel_pair, "reason": f"preprocess error: {exc}"})
            log(f"[skip] {rel_pair} – preprocess error: {exc}")
            continue

        missing_doc = ensure_doc_types(df_processed)
        if missing_doc:
            skipped_rows.append({"pair_path": rel_pair, "reason": missing_doc})
            log(f"[skip] {rel_pair} – {missing_doc}")
            continue

        try:
            counts_df, delta_df = compute_counts(df_processed, categories, stemmed_dict)
        except Exception as exc:
            skipped_rows.append({"pair_path": rel_pair, "reason": f"count error: {exc}"})
            log(f"[skip] {rel_pair} – count error: {exc}")
            continue

        overunit_id = derive_overunit_id(pair_dir)
        counts_df = counts_df.reset_index().rename(columns={"index": "doc_type"})
        counts_df["doc_name"] = counts_df["doc_type"].apply(lambda dt: df.loc[df["doc_type"] == dt, "doc_name"].iloc[0])
        counts_df["doc_path"] = counts_df["doc_type"].apply(
            lambda dt: df.loc[df["doc_type"] == dt, "doc_path"].iloc[0]
        )
        counts_df["pair_relative_path"] = rel_pair
        counts_df["overunit_id"] = overunit_id
        counts_df["dictionary_hash"] = dictionary_hash
        counts_df["dictionary_path"] = relative_path_or_str(dict_path, root)
        counts_df["dictionary_term_count"] = dictionary_term_count

        delta_df["pair_relative_path"] = rel_pair
        delta_df["results_path"] = relative_path_or_str(results_dir, root)
        delta_df["overunit_id"] = overunit_id
        delta_df["dictionary_hash"] = dictionary_hash
        delta_df["dictionary_path"] = relative_path_or_str(dict_path, root)
        delta_df["dictionary_term_count"] = dictionary_term_count
        delta_df["ge_doc"] = df.loc[df["doc_type"] == "GE", "doc_path"].iloc[0]
        delta_df["lr_doc"] = df.loc[df["doc_type"] == "LR", "doc_path"].iloc[0]

        cols_order = [
            "doc_type",
            "doc_name",
            "doc_path",
            "pair_relative_path",
            "overunit_id",
            "dictionary_path",
            "dictionary_hash",
            "dictionary_term_count",
        ] + list(categories) + ["total_dict_matches"]
        counts_df = counts_df[cols_order]

        delta_cols = [
            "pair_relative_path",
            "results_path",
            "overunit_id",
            "dictionary_path",
            "dictionary_hash",
            "dictionary_term_count",
            "ge_doc",
            "lr_doc",
        ] + [f"delta_{cat}" for cat in categories] + ["delta_total_dict_matches"]
        delta_df = delta_df[delta_cols]

        create_results_dir(results_dir, args.dry_run)
        doc_summary_path = results_dir / "doc_summary_pair.csv"
        delta_summary_path = results_dir / "delta_summary.csv"

        write_csv(doc_summary_path, counts_df, args.dry_run)
        write_csv(delta_summary_path, delta_df, args.dry_run)

        summary_rows.append(delta_df)

        delta_total = delta_df["delta_total_dict_matches"].iloc[0]
        log(f"[ok] {rel_pair} – Δ_total={delta_total:+d}")

    if progress_bar is not None:
        progress_bar.close()
        progress_bar = None

    if summary_rows:
        combined_summary = pd.concat(summary_rows, ignore_index=True)
        merge_csv(out_global, combined_summary, key_columns=["pair_relative_path"], dry_run=args.dry_run)

    if skipped_rows:
        skipped_df = pd.DataFrame(skipped_rows)
        merge_csv(skipped_log_path, skipped_df, key_columns=["pair_path"], dry_run=args.dry_run)

    if args.verbose:
        log(f"Processed pairs: {len(summary_rows)}")
        log(f"Skipped pairs:  {len(skipped_rows)}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
