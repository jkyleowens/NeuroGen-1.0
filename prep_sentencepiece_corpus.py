#!/usr/bin/env python3
import argparse, csv, glob, io, json, os, re, sys, unicodedata
from html import unescape

URL_RE = re.compile(r"""https?://\S+|www\.\S+""", re.IGNORECASE)
EMAIL_RE = re.compile(r"""\b[\w.\-+%]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b""")
CTRL_RE = re.compile(r"[\u0000-\u0008\u000B-\u000C\u000E-\u001F\u007F]")
MULTI_WS_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

def normalize_text(s, strip_html=True, remove_urls=True, remove_emails=True,
                   lowercase=False, preserve_newlines=False, mode="text"):
    if not isinstance(s, str):
        return ""
    # Decode HTML entities
    s = unescape(s)
    # Optionally strip tags (simple heuristic; do BEFORE removing URLs)
    if strip_html:
        s = HTML_TAG_RE.sub(" ", s)
    # Unicode normalize
    s = unicodedata.normalize("NFKC", s)
    # Remove URLs/emails
    if remove_urls:
        s = URL_RE.sub(" ", s)
    if remove_emails:
        s = EMAIL_RE.sub(" ", s)
    # Remove control chars
    s = CTRL_RE.sub(" ", s)
    # Collapse whitespace
    if preserve_newlines:
        # collapse runs of spaces/tabs but keep newlines
        s = re.sub(r"[^\S\r\n]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
    else:
        s = MULTI_WS_RE.sub(" ", s)
    s = s.strip()
    if lowercase and mode != "code":
        s = s.lower()
    return s

def yield_lines_from_file(path, args):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md", ".rst", ".log"]:
        with io.open(path, "r", encoding=args.encoding, errors="ignore") as f:
            for raw in f:
                text = normalize_text(raw, strip_html=args.strip_html, remove_urls=not args.keep_urls,
                                      remove_emails=not args.keep_emails, lowercase=args.lowercase,
                                      preserve_newlines=False, mode=args.mode)
                if text:
                    yield from split_and_filter(text, args)
    elif ext in [".jsonl", ".jsonl.zst", ".jsonl.gz", ".json"]:
        # naive: read as jsonl if possible; if plain json list, iterate
        opener = open
        if ext == ".json":
            with io.open(path, "r", encoding=args.encoding, errors="ignore") as f:
                try:
                    obj = json.load(f)
                    if isinstance(obj, list):
                        for item in obj:
                            text = extract_from_json(item, args)
                            if text:
                                yield from split_and_filter(
                                    normalize_text(text, strip_html=args.strip_html,
                                                   remove_urls=not args.keep_urls,
                                                   remove_emails=not args.keep_emails,
                                                   lowercase=args.lowercase,
                                                   preserve_newlines=False,
                                                   mode=args.mode),
                                    args)
                    else:
                        text = extract_from_json(obj, args)
                        if text:
                            yield from split_and_filter(normalize_text(text, strip_html=args.strip_html,
                                                                       remove_urls=not args.keep_urls,
                                                                       remove_emails=not args.keep_emails,
                                                                       lowercase=args.lowercase,
                                                                       preserve_newlines=False,
                                                                       mode=args.mode),
                                                       args)
                except Exception:
                    pass
        else:
            # Treat as jsonl; user can pre-decompress if needed
            with io.open(path, "r", encoding=args.encoding, errors="ignore") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    text = extract_from_json(obj, args)
                    if text:
                        text = normalize_text(text, strip_html=args.strip_html,
                                              remove_urls=not args.keep_urls,
                                              remove_emails=not args.keep_emails,
                                              lowercase=args.lowercase, preserve_newlines=False,
                                              mode=args.mode)
                        if text:
                            yield from split_and_filter(text, args)
    elif ext in [".csv", ".tsv"]:
        dialect = "excel" if ext == ".csv" else "excel-tab"
        with io.open(path, "r", encoding=args.encoding, errors="ignore", newline="") as f:
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                text = extract_from_row(row, args)
                if text:
                    text = normalize_text(text, strip_html=args.strip_html,
                                          remove_urls=not args.keep_urls,
                                          remove_emails=not args.keep_emails,
                                          lowercase=args.lowercase, preserve_newlines=False,
                                          mode=args.mode)
                    if text:
                        yield from split_and_filter(text, args)
    else:
        # default: treat as text
        with io.open(path, "r", encoding=args.encoding, errors="ignore") as f:
            for raw in f:
                text = normalize_text(raw, strip_html=args.strip_html, remove_urls=not args.keep_urls,
                                      remove_emails=not args.keep_emails, lowercase=args.lowercase,
                                      preserve_newlines=False, mode=args.mode)
                if text:
                    yield from split_and_filter(text, args)

def extract_from_json(obj, args):
    # Find text fields by priority: explicit keys -> common fallbacks
    keys = args.json_text_keys or []
    fallbacks = ["text", "content", "body", "paragraph", "article", "message"]
    for k in keys + fallbacks:
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], str):
            return obj[k]
    # Heuristic: join all string fields (short) if no explicit text field
    if isinstance(obj, dict):
        parts = [v for v in obj.values() if isinstance(v, str)]
        if parts:
            return " ".join(parts)
    return None

def extract_from_row(row, args):
    if args.csv_text_cols:
        cols = [c for c in args.csv_text_cols if c in row]
        if cols:
            return " ".join(row[c] or "" for c in cols)
    # else join all string columns
    parts = [v for v in row.values() if isinstance(v, str)]
    return " ".join(parts) if parts else None

def split_and_filter(text, args):
    if args.mode == "code" or args.no_sent_split:
        lines = [text]
    else:
        # simple sentence split; safe default
        lines = SENT_SPLIT_RE.split(text)
    for s in lines:
        s = s.strip()
        if not s:
            continue
        if args.min_len and len(s) < args.min_len:
            continue
        if args.max_len and len(s) > args.max_len:
            # optionally chunk very long lines
            if args.chunk_long:
                for i in range(0, len(s), args.max_len):
                    piece = s[i:i+args.max_len].strip()
                    if len(piece) >= args.min_len:
                        yield piece
            continue
        yield s

def iter_input_paths(inputs):
    for p in inputs:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    yield os.path.join(root, fn)
        else:
            if any(ch in p for ch in "*?[]"):
                for m in glob.glob(p):
                    yield m
            else:
                yield p

def main():
    ap = argparse.ArgumentParser(description="Prepare data.txt for SentencePiece.")
    ap.add_argument("inputs", nargs="+", help="Files/dirs/globs of TXT/JSON/JSONL/CSV.")
    ap.add_argument("--out", default="data.txt", help="Output file (one example per line).")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--json-text-keys", nargs="*", default=[],
                    help="JSON keys to use for text (priority order).")
    ap.add_argument("--csv-text-cols", nargs="*", default=[],
                    help="CSV column names to concatenate for text.")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase (not recommended for code).")
    ap.add_argument("--keep-urls", action="store_true", help="Keep URLs (default removes).")
    ap.add_argument("--keep-emails", action="store_true", help="Keep emails (default removes).")
    ap.add_argument("--strip-html", action="store_true", help="Strip HTML tags/entities first.")
    ap.add_argument("--min-len", type=int, default=10, help="Drop lines shorter than this.")
    ap.add_argument("--max-len", type=int, default=1000, help="Drop/split lines longer than this.")
    ap.add_argument("--chunk-long", action="store_true", help="Chunk long lines to max-len windows.")
    ap.add_argument("--no-sent-split", action="store_true", help="Don’t split into sentences.")
    ap.add_argument("--dedupe", action="store_true", help="Deduplicate lines.")
    ap.add_argument("--mode", choices=["text","code"], default="text",
                    help="‘code’ skips lowercasing and sentence split heuristics.")
    args = ap.parse_args()

    seen = set()
    kept = 0
    with io.open(args.out, "w", encoding="utf-8") as out_f:
        for path in iter_input_paths(args.inputs):
            try:
                for line in yield_lines_from_file(path, args):
                    if args.dedupe:
                        h = hash(line)
                        if h in seen: 
                            continue
                        seen.add(h)
                    out_f.write(line + "\n")
                    kept += 1
            except Exception as e:
                print(f"[warn] failed {path}: {e}", file=sys.stderr)
    print(f"[ok] wrote {kept} lines to {args.out}")

if __name__ == "__main__":
    main()
