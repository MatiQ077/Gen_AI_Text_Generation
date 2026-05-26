import argparse
import csv
import io
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import fitz
import pandas as pd
import requests
from tqdm import tqdm


DEFAULT_INPUT_PATH = Path("data/pdf_guides.csv")
DEFAULT_OUTPUT_PATH = Path("data/raw/pdf_guides.jsonl")
DEFAULT_FAILURES_PATH = Path("data/raw/pdf_extraction_failures.csv")
DEFAULT_KEEP_PDF_DIR = Path("data/raw/pdf_guides")

HEADERS = {"User-Agent": "TravelAssistantStudentProject/1.0"}

def basic_clean(text: str) -> str:   
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_filename(filename: str) -> str:
   
    filename = str(filename).strip()
    filename = filename.replace(" ", "_")
    filename = re.sub(r"[^A-Za-z0-9_.-]", "", filename)

    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return filename

def download_pdf_bytes(url: str, timeout: int = 60) -> Optional[bytes]:
    
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=timeout,
            stream=True,
            allow_redirects=True,
        )

        response.raise_for_status()
        content = response.content

        if not content.startswith(b"%PDF"):
            print(f"[WARNING] Downloaded content may not be a valid PDF: {url}")
            return None

        return content

    except requests.RequestException as error:
        print(f"[ERROR] Could not download {url}: {error}")
        return None

def save_pdf_bytes(pdf_bytes: bytes, output_path: Path) -> None:
   
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        file.write(pdf_bytes)

def extract_pages_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict]:
    
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = basic_clean(text)

            pages.append({
                "page_number": page_number,
                "text": text,
                "word_count": len(text.split()),
            })

    return pages

def split_text_into_chunks(
    text: str,
    max_words: int = 200,
    min_words: int = 60,
    overlap_words: int = 0,
) -> List[str]:
    
    words = text.split()

    if not words:
        return []

    chunks = []
    step = max_words - overlap_words

    if step <= 0:
        raise ValueError("overlap_words must be smaller than max_words.")

    for start in range(0, len(words), step):
        chunk_words = words[start:start + max_words]

        if len(chunk_words) < min_words:
            continue

        chunk = " ".join(chunk_words).strip()
        chunks.append(chunk)

    return chunks

def looks_like_good_text(text: str) -> bool:
   
    words = text.split()

    if len(words) < 60:
        return False

    #Checking if a sentence has any sentence mark
    sentence_marks = text.count(".") + text.count("!") + text.count("?")
    if sentence_marks < 2:
        return False

    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.25:
        return False

    # Filter chunks that are mostly uppercase, often menus/maps/headers.
    letters = [char for char in text if char.isalpha()]
    if letters:
        uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
        if uppercase_ratio > 0.65:
            return False

    return True

def write_jsonl_record(record: Dict, output_path: Path) -> None:
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_failures(failures: List[Dict], failures_path: Path) -> None:
    
    if not failures:
        return

    failures_path.parent.mkdir(parents=True, exist_ok=True)

    with failures_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "url", "reason"],
        )
        writer.writeheader()
        writer.writerows(failures)

def process_pdf_row(
    row: pd.Series,
    output_path: Path,
    keep_pdfs: bool,
    keep_pdf_dir: Path,
    max_words: int,
    min_words: int,
    overlap_words: int,
) -> tuple[int, Optional[Dict]]:
    
    url = str(row.get("url", "")).strip()
    filename = clean_filename(row.get("filename", "unknown.pdf"))

    if not url or url.lower() == "nan":
        return 0, {
            "filename": filename,
            "url": url,
            "reason": "Missing URL",
        }

    pdf_bytes = download_pdf_bytes(url)

    if pdf_bytes is None:
        return 0, {
            "filename": filename,
            "url": url,
            "reason": "Download failed or invalid PDF",
        }

    if keep_pdfs:
        save_pdf_bytes(pdf_bytes, keep_pdf_dir / filename)

    try:
        pages = extract_pages_from_pdf_bytes(pdf_bytes)
    except Exception as error:
        return 0, {
            "filename": filename,
            "url": url,
            "reason": f"PDF extraction error: {error}",
        }

    total_words = sum(page["word_count"] for page in pages)

    if total_words < 300:
        return 0, {
            "filename": filename,
            "url": url,
            "reason": f"Too little extracted text: {total_words} words. Possibly scanned/image-based PDF.",
        }

    saved_chunks = 0

    for page in pages:
        chunks = split_text_into_chunks(
            page["text"],
            max_words=max_words,
            min_words=min_words,
            overlap_words=overlap_words,
        )

        for chunk_id, chunk in enumerate(chunks):
            if not looks_like_good_text(chunk):
                continue

            record = {
                "source": "pdf_guide",
                "page_type": "travel_guide_pdf",
                "url": url,
                "country": row.get("country", ""),
                "city": row.get("city", ""),
                "title": filename,
                "section": f"page_{page['page_number']}",
                "category": row.get("category", "general"),
                "source_name": row.get("source_name", ""),
                "page_number": page["page_number"],
                "chunk_id": chunk_id,
                "text": chunk,
            }

            write_jsonl_record(record, output_path)
            saved_chunks += 1

    if saved_chunks == 0:
        return 0, {
            "filename": filename,
            "url": url,
            "reason": "No clean chunks passed filtering",
        }

    return saved_chunks, None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PDF travel guides, extract text, and save directly to JSONL."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to CSV file with PDF metadata and URLs.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to output JSONL file.",
    )

    parser.add_argument(
        "--failures-output",
        type=str,
        default=str(DEFAULT_FAILURES_PATH),
        help="Path to CSV file for failed downloads/extractions.",
    )

    parser.add_argument(
        "--keep-pdfs",
        action="store_true",
        help="Save original downloaded PDFs to data/raw/pdf_guides.",
    )

    parser.add_argument(
        "--keep-pdf-dir",
        type=str,
        default=str(DEFAULT_KEEP_PDF_DIR),
        help="Directory for saved PDFs when --keep-pdfs is used.",
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=200,
        help="Maximum words per extracted chunk.",
    )

    parser.add_argument(
        "--min-words",
        type=int,
        default=60,
        help="Minimum words per extracted chunk.",
    )

    parser.add_argument(
        "--overlap-words",
        type=int,
        default=0,
        help="Word overlap between chunks. Recommended 0 for training.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Delay between PDF downloads in seconds.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of PDFs to process.",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of replacing it.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    failures_path = Path(args.failures_output)
    keep_pdf_dir = Path(args.keep_pdf_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input metadata file not found: {input_path}")

    metadata_df = pd.read_csv(input_path)

    required_columns = {"url"}
    missing_columns = required_columns - set(metadata_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in metadata CSV: {missing_columns}")

    if "filename" not in metadata_df.columns:
        metadata_df["filename"] = [f"pdf_guide_{i + 1}.pdf" for i in range(len(metadata_df))]

    if args.limit is not None:
        metadata_df = metadata_df.head(args.limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.append:
        output_path.unlink()

    if failures_path.exists():
        failures_path.unlink()

    total_saved_chunks = 0
    failures = []

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing PDF guides"):
        saved_chunks, failure = process_pdf_row(
            row=row,
            output_path=output_path,
            keep_pdfs=args.keep_pdfs,
            keep_pdf_dir=keep_pdf_dir,
            max_words=args.max_words,
            min_words=args.min_words,
            overlap_words=args.overlap_words,
        )

        total_saved_chunks += saved_chunks

        if failure is not None:
            failures.append(failure)

        time.sleep(args.sleep)

    save_failures(failures, failures_path)

    print("\nPDF extraction summary")
    print("-" * 40)
    print(f"PDFs processed: {len(metadata_df)}")
    print(f"Saved chunks: {total_saved_chunks}")
    print(f"Failed/skipped PDFs: {len(failures)}")
    print(f"Output JSONL: {output_path}")

    if args.keep_pdfs:
        print(f"Saved PDFs folder: {keep_pdf_dir}")

    if failures:
        print(f"Failures logged to: {failures_path}")

if __name__ == "__main__":
    main()