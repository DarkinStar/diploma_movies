import argparse
import csv
import re
from pathlib import Path


WHITESPACE_RE = re.compile(r"\s+")
TRAILING_CITATION_RE = re.compile(r"(?<=[A-Za-z])\.(\d+)$")
TRAILING_NOTE_RE = re.compile(r"\[(\d+|note \d+)\]", re.IGNORECASE)
PART_NUM_RE = re.compile(r"\bpart\s+([ivx]+|\d+)\b", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Clean the movie plots dataset.")
    parser.add_argument(
        "--input",
        default="datasets/cleaned_movie_plots.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="datasets/cleaned_movie_plots_v2.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = TRAILING_NOTE_RE.sub("", cleaned)
    cleaned = TRAILING_CITATION_RE.sub(".", cleaned)
    cleaned = cleaned.replace("â€”", "-").replace("â€“", "-")
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def clean_genre(genre: str) -> str:
    cleaned = normalize_whitespace((genre or "").strip().lower())
    return cleaned or "unknown"


def canonicalize_title(title: str) -> str:
    cleaned = normalize_whitespace((title or "").strip().lower())
    cleaned = cleaned.replace("harry potter and the philosopher's stone", "harry potter and the sorcerer's stone")
    cleaned = PART_NUM_RE.sub(
        lambda match: f"part {roman_to_int(match.group(1).lower())}",
        cleaned,
    )
    return cleaned


def roman_to_int(value: str) -> str:
    if value.isdigit():
        return value
    mapping = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
    return str(mapping.get(value, value))


def build_embedding_text(row: dict) -> str:
    parts = [f"Title: {row['title']}"]
    if row["genre"] != "unknown":
        parts.append(f"Genre: {row['genre']}")
    parts.append(f"Summary: {row['summary']}")
    parts.append(f"Plot: {row['plot']}")
    return " ".join(parts)


def is_suspicious(row: dict) -> bool:
    plot = row["plot"]
    summary = row["summary"]
    if len(plot) < 80 or len(summary) < 40:
        return True
    if "the story as told by moving picture world reads" in plot and len(summary) < 120:
        return True
    return False


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def save_rows(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    fieldnames, rows = load_rows(input_path)
    if not fieldnames:
        raise ValueError("Input CSV has no header.")

    cleaned_rows = []
    seen_keys = set()

    for row in rows:
        cleaned_row = {
            "title": normalize_whitespace((row.get("title") or "").strip()),
            "genre": clean_genre(row.get("genre") or ""),
            "plot": clean_text(row.get("plot") or ""),
            "summary": clean_text(row.get("summary") or ""),
            "release_year": normalize_whitespace((row.get("release_year") or "").strip()),
        }

        dedupe_key = (canonicalize_title(cleaned_row["title"]), cleaned_row["release_year"])
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        cleaned_row["embedding_text"] = build_embedding_text(cleaned_row)
        cleaned_row["suspicious_row"] = "true" if is_suspicious(cleaned_row) else "false"
        cleaned_rows.append(cleaned_row)

    output_fields = ["title", "genre", "plot", "summary", "release_year", "embedding_text", "suspicious_row"]
    save_rows(output_path, output_fields, cleaned_rows)

    suspicious_count = sum(1 for row in cleaned_rows if row["suspicious_row"] == "true")
    print(f"Saved {len(cleaned_rows)} rows to {output_path}")
    print(f"Flagged {suspicious_count} suspicious rows for manual review")


if __name__ == "__main__":
    main()
