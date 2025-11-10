#!/usr/bin/env python3
"""
Batch_Cosine_Similarity_v6

Walk a root directory containing many subfolders. Each subfolder is expected to
contain exactly two TXT files (plain-text OCR outputs). For each subfolder, read the text,
run the same preprocessing pipeline as in Notebooks/Batch_Cosine_Similarity_v1.ipynb, build
TF-IDF vectors, compute cosine similarity, and save a long-form CSV in that
subfolder (default: cosine_similarity.csv).

Usage examples:
  python Batch_Cosine_Similarity_v6.py /path/to/root
  python Batch_Cosine_Similarity_v6.py /path/to/root --include-self --all-pairs
  python Batch_Cosine_Similarity_v6.py /path/to/root --recursive

Notes:
  - Requires: nltk, scikit-learn, pandas
  - Expects NLTK data available locally. This script appends './nltk_data' from
    the repository root to nltk.data.path. If not found, tokenization/stopwords
    may raise a LookupError.
"""

from __future__ import annotations

import argparse
import os
import re
import string
import unicodedata
from glob import glob
from typing import List, Tuple


# --- Helper: extract federal state and legislative period from a folder path ---
def extract_state_wp(path_str: str):
    """Return (state, legislative_period) parsed from a directory path.
    Looks for a path segment like '09. WP' (case-insensitive) and returns that as legislative_period,
    with the immediately preceding segment as state, e.g., 'Bayern'.
    If not found, returns (None, None).
    """
    try:
        norm = os.path.normpath(path_str)
        parts = norm.split(os.sep)
        wp = None
        state = None
        wp_re = re.compile(r"^\s*\d{1,2}\.\s*WP\s*$", re.IGNORECASE)
        for i, seg in enumerate(parts):
            if wp_re.match(seg):
                wp = seg.strip()
                state = parts[i-1] if i-1 >= 0 else None
                break
        return state, wp
    except Exception:
        return None, None

# --- Helper: convert full state name to two-letter abbreviation ---
def get_state_abbr(state: str | None) -> str | None:
    if not state:
        return None
    # Normalize Unicode (fixes composed vs. decomposed umlauts like "Thüringen")
    def _nfc(s: str) -> str:
        return unicodedata.normalize("NFC", s.strip())
    # Remove diacritics for robust ASCII fallback (e.g., "Thuringen")
    def _strip_diacritics(s: str) -> str:
        return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
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
    s = _nfc(state)
    # Try exact normalized match first
    if s in mapping:
        return mapping[s]
    # Try diacritic-stripped ASCII form (e.g., "Thuringen")
    s_ascii = _strip_diacritics(s)
    if s_ascii in mapping:
        return mapping[s_ascii]
    # As a last resort, return the original normalized string
    return s
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _append_local_nltk_path():
    nltk_data_dir = "/Users/jannicksteinacker/NextCloud/Individuell/Forschung (persönlich)/Dissertation/2025/Paperidee I – Impact of Gender on Climate Policy/Analysis/Text-as-Data/Text Analysis in Python/nltk_data"  
    if os.path.isdir(nltk_data_dir):
        nltk.data.path = [nltk_data_dir] + nltk.data.path
    else:
        print(f"⚠️  nltk_data not found at {nltk_data_dir}")


_append_local_nltk_path()

def clean_latex_artifacts(text: str) -> str:
    """
    Clean LaTeX math remnants and preserve semantic content.
    Examples:
      - "$\\mathrm{CO}{2}$" → "CO2"
      - "$65 \\mathrm{~dB}(\\mathrm{~A})$" → "65 dB(A)"
      - "$\\$ 9$" → "§ 9"
      - Removes $$...$$ and \\begin...\\end...
    """
    # Remove math delimiters and environments
    text = re.sub(r"\$\$|\$", " ", text)
    text = re.sub(r"\\begin\{.*?\}|\\end\{.*?\}", " ", text)

    # Replace escaped LaTeX paragraph symbol \$ with §
    text = text.replace(r"\$", "§")

    # Unwrap \text{} or \mathrm{} and keep content
    text = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", text)

    # Remove formatting helpers like \left, \right, \int, etc.
    text = re.sub(r"\\(left|right|int|gathered|text|mathrm)\b", " ", text)

    # Fix parentheses that were smashed against words: db(Umwelt) → db (Umwelt)
    text = re.sub(r"([a-zA-Z])\(", r"\1 (", text)

    # Collapse excessive spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---- Preprocessing functions (mirrors the notebook) ----
def fix_linebreak_hyphens(text: str) -> str:
    # Merge words split across a line: hyphen followed by space(s)
    return re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)


def text_lowercase(text: str) -> str:
    return text.lower()


def remove_urls(text: str) -> str:
    # Match typical URL patterns (http, https, www)
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_symbols(text: str) -> str:
    # Keep letters (incl. umlauts), digits, whitespace, hyphens
    return re.sub(r"[^0-9A-Za-zÄÖÜäöüß\-\s]", " ", text)


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", " ", text)


def read_text_file(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def _ensure_tokenizer_available():
    # Try using tokenizer; if missing, raise clear message
    try:
        _ = word_tokenize("probe")
    except LookupError as e:
        raise LookupError(
            "NLTK tokenizer data (punkt) not found. Please download it to your local 'nltk_data' folder.\n"
            "In Python: import nltk; nltk.download('punkt', download_dir='nltk_data')"
        ) from e


def _load_stopwords() -> set:
    try:
        return set(stopwords.words("german"))
    except LookupError as exc:
        raise LookupError(
            "NLTK stopwords corpus (stopwords) not found. Please download it to your local 'nltk_data' folder.\n"
            "In Python: import nltk; nltk.download('stopwords', download_dir='nltk_data')"
        ) from exc


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # text_clean pipeline
    df["text_clean"] = df["text"].apply(clean_latex_artifacts)
    df["text_clean"] = df["text_clean"].apply(fix_linebreak_hyphens)
    df["text_clean"] = df["text_clean"].apply(text_lowercase)
    df["text_clean"] = df["text_clean"].apply(remove_urls)
    df["text_clean"] = df["text_clean"].apply(remove_symbols)
    # df["text_clean"] = df["text_clean"].apply(remove_numbers)  # removed as per instructions

    # tokenization
    _ensure_tokenizer_available()
    df["tokens"] = df["text_clean"].apply(word_tokenize)

    # remove punctuation tokens and single letters
    punct_set = set(string.punctuation)
    df["tokens"] = df["tokens"].apply(lambda toks: [t for t in toks if t not in punct_set])
    df["tokens"] = df["tokens"].apply(lambda toks: [t for t in toks if len(t) > 1])

    # unwanted tokens (from notebook)
    def is_double_letter_token(t):
        return len(t) == 2 and t[0].isalpha() and t[0].lower() == t[1].lower()
    df['tokens'] = df['tokens'].apply(lambda toks: [t for t in toks if not is_double_letter_token(t)])

    # stopwords (German)
    stop_words = _load_stopwords()
    df["tokens"] = df["tokens"].apply(lambda toks: [t for t in toks if t.lower() not in stop_words])

    # stemming (German)
    stemmer = SnowballStemmer("german")
    df["tokens"] = df["tokens"].apply(lambda toks: [stemmer.stem(t) for t in toks])

    # join back for vectorizers
    df["text_processed"] = df["tokens"].apply(lambda toks: " ".join(toks))
    return df


def tfidf_cosine(corpus_df: pd.DataFrame) -> pd.DataFrame:
    vec = TfidfVectorizer()
    X = vec.fit_transform(corpus_df["text_processed"])
    sim = cosine_similarity(X)
    names = corpus_df["law_name"].tolist()
    sim_df = pd.DataFrame(sim, index=names, columns=names)
    return sim_df


def longform(sim_df: pd.DataFrame, include_self: bool, unique_pairs: bool) -> pd.DataFrame:
    lf = sim_df.stack().reset_index()
    lf.columns = ["doc_a", "doc_b", "cosine_similarity"]
    if not include_self:
        lf = lf[lf["doc_a"] != lf["doc_b"]]
    if unique_pairs:
        # Keep one row per unordered pair: enforce doc_a < doc_b lexicographically
        lf = lf[lf.apply(lambda r: r["doc_a"] < r["doc_b"], axis=1)]
    return lf.reset_index(drop=True)


def process_folder(folder_path: str, out_csv_name: str, include_self: bool, unique_pairs: bool) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    # Prefer OCR outputs in a 'Rohtexte' subfolder; fall back to '*.txt' in the folder
    roh = os.path.join(folder_path, "Rohtexte")
    search_dir = roh if os.path.isdir(roh) else folder_path
    txt_paths = sorted(glob(os.path.join(search_dir, "*.txt")))
    if len(txt_paths) != 2:
        raise ValueError(f"Expected exactly 2 TXTs in {search_dir}, found {len(txt_paths)}")

    data = []
    for p in txt_paths:
        data.append({
            "doc_id": p,
            "law_name": os.path.splitext(os.path.basename(p))[0],
            "text": read_text_file(p),
        })
    df = pd.DataFrame(data)
    df = preprocess_dataframe(df)
    sim_df = tfidf_cosine(df)
    lf = longform(sim_df, include_self=include_self, unique_pairs=unique_pairs)

    out_csv = os.path.join(folder_path, out_csv_name)
    # Compute OVERUNIT_ID from path-derived metadata
    state, legislative_period = extract_state_wp(folder_path)
    abbr = get_state_abbr(state)
    if abbr and legislative_period:
        # Remove all non-digits from legislative_period (e.g., "09. WP" -> "09")
        overunit_id = abbr + re.sub(r"\D", "", legislative_period)
    else:
        overunit_id = None

    lf["OVERUNIT_ID"] = overunit_id
    # Put OVERUNIT_ID in front
    cols = ["OVERUNIT_ID"] + [c for c in lf.columns if c != "OVERUNIT_ID"]
    lf = lf[cols]

    lf.to_csv(out_csv, index=False)
    return sim_df, lf, out_csv


def iterate_subfolders(root_dir: str, recursive: bool) -> List[str]:
    """
    Return folders to process exactly once:
    - If a folder contains a 'Rohtexte' subfolder, include the *parent* folder (not 'Rohtexte').
    - Else, if a folder itself contains .txt files, include that folder.
    - Never include 'Rohtexte' itself as a processing target and do not descend into it.
    """
    subdirs: List[str] = []
    root_dir = os.path.abspath(root_dir)

    if recursive:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Never descend into 'Rohtexte' (prevents duplicate processing)
            if "Rohtexte" in dirnames:
                # Include the parent law folder once
                subdirs.append(dirpath)
                # Remove 'Rohtexte' from traversal so we don't visit it later
                dirnames[:] = [d for d in dirnames if d != "Rohtexte"]
                # Continue walking other subfolders (e.g., sibling law folders)
                continue

            # If no 'Rohtexte' subfolder, include folder if it has .txt files directly
            if any(fn.lower().endswith(".txt") for fn in filenames):
                subdirs.append(dirpath)

        # Deduplicate while preserving order
        seen = set()
        unique_subdirs = []
        for p in subdirs:
            if p not in seen:
                unique_subdirs.append(p)
                seen.add(p)
        return unique_subdirs
    else:
        # Non-recursive: only immediate subdirectories of root
        out = []
        for entry in sorted(os.scandir(root_dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            # If the immediate child has a 'Rohtexte' subfolder, include the child
            roh = os.path.join(entry.path, "Rohtexte")
            if os.path.isdir(roh):
                out.append(entry.path)
                continue
            # Else include if it has .txt directly
            if any(name.lower().endswith(".txt") for name in os.listdir(entry.path)):
                out.append(entry.path)
        return out


def main():
    parser = argparse.ArgumentParser(description="Compute TF-IDF cosine similarity for two-TXT folders.")
    parser.add_argument("root_dir", help="Root directory containing subfolders with exactly two TXT files (e.g., in 'Rohtexte').")
    parser.add_argument("--out-csv-name", default="cosine_similarity.csv", help="Output CSV name per folder.")
    parser.add_argument("--include-self", action="store_true", help="Include self-similarity rows in CSV.")
    parser.add_argument("--all-pairs", dest="unique_pairs", action="store_false", help="Keep both (A,B) and (B,A).")
    parser.add_argument("--recursive", action="store_true", help="Recurse into nested subfolders; process each law folder once (parent of 'Rohtexte' or folders with .txt files).")
    args = parser.parse_args()

    root = os.path.abspath(args.root_dir)
    if not os.path.isdir(root):
        raise SystemExit(f"Not a directory: {root}")

    subdirs = iterate_subfolders(root, recursive=args.recursive)
    if not subdirs:
        print(f"No subfolders found under: {root}")
        return

    summary_rows = []
    processed = 0
    skipped = 0
    skip_rows = []

    for folder in subdirs:
        try:
            roh = os.path.join(folder, "Rohtexte")
            search_dir = roh if os.path.isdir(roh) else folder
            txts = sorted(glob(os.path.join(search_dir, "*.txt")))
            if len(txts) != 2:
                skipped += 1
                print(f"[skip] {folder} – expected 2 TXTs in {search_dir}, found {len(txts)}")
                skip_rows.append({
                    "folder": folder,
                    "search_dir": search_dir,
                    "txt_count": len(txts),
                    "reason": f"Expected 2 TXTs, found {len(txts)}"
                })
                continue

            print(f"[proc] {folder} (source: {search_dir})")
            _, lf, out_csv = process_folder(
                folder,
                out_csv_name=args.out_csv_name,
                include_self=args.include_self,
                unique_pairs=args.unique_pairs,
            )

            # For summary, keep one row per folder (the unique pair), or first row otherwise
            if not args.include_self and args.unique_pairs and not lf.empty:
                row = lf.iloc[0]
                summary_rows.append({
                    "folder": folder,
                    "OVERUNIT_ID": (lambda _s, _w: ((get_state_abbr(_s) + re.sub(r"\D", "", _w)) if get_state_abbr(_s) and _w else None))(*extract_state_wp(folder)),
                    "doc_a": row["doc_a"],
                    "doc_b": row["doc_b"],
                    "cosine_similarity": float(row["cosine_similarity"]),
                    "csv_path": out_csv,
                })
            processed += 1
        except Exception as e:
            skipped += 1
            print(f"[error] {folder}: {e}")
            skip_rows.append({
                "folder": folder,
                "search_dir": search_dir if 'search_dir' in locals() else None,
                "txt_count": len(txts) if 'txts' in locals() else None,
                "reason": f"error: {e}"
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(root, "cosine_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path} ({len(summary_rows)} rows)")

    if skip_rows:
        skip_log_path = os.path.join(root, "cosine_skipped.csv")
        pd.DataFrame(skip_rows).to_csv(skip_log_path, index=False)
        print(f"Saved skipped summary: {skip_log_path} ({len(skip_rows)} rows)")

    print(f"Done. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
