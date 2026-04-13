#!/usr/bin/env python3
"""
language_filter.py
==================
Filter JSONL files to keep only English only rows using fasttext
with confidence thresholding, chunked detection for mixed-language documents,
byte-range file splitting for large files, and multiprocessing worker pool.

Input: CSV file with (path, key) columns — path can be folder or file.
Output: per-file _en.jsonl, _non_selected.jsonl, _unknown.jsonl,
        manifest CSV, manifest JSON.

Usage:
    python language_filter.py \\
        --file-list inputs.csv \\
        --output /data/filtered \\
        --workers 16 \\
        --split-threshold 512 \\
        --confidence 0.7 \\
        --default-key messages

Requirements:
    pip install fasttext-wheel --break-system-packages
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
"""

import argparse
import csv
import os
import sys
import time
import logging
import random
import shutil
import hashlib
import socket
import multiprocessing as mp
import threading

try:
    import orjson
    def json_loads(s):
        return orjson.loads(s)
    def json_dumps(obj):
        return orjson.dumps(obj).decode("utf-8")
    _JSON_LIB = "orjson"
except ImportError:
    import json
    def json_loads(s):
        return json.loads(s)
    def json_dumps(obj):
        return json.dumps(obj, ensure_ascii=False)
    _JSON_LIB = "json"

import json  # always needed for json.dump in manifest writing
from pathlib import Path
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: str, run_timestamp: str) -> logging.Logger:
    logger = logging.getLogger("language_filter")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Log to current working directory logs/
    cwd_log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(cwd_log_dir, exist_ok=True)
    cwd_log = os.path.join(cwd_log_dir, f"lang_filter_run_{run_timestamp}.log")
    fh_cwd = logging.FileHandler(cwd_log)
    fh_cwd.setFormatter(formatter)
    logger.addHandler(fh_cwd)

    # Log to output/logs/run_timestamp/
    log_dir = os.path.join(output_dir, "logs", f"lang_filter_run_{run_timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    fh_out = logging.FileHandler(os.path.join(log_dir, "language_filter.log"))
    fh_out.setFormatter(formatter)
    logger.addHandler(fh_out)

    logger.info(f"Logs: {cwd_log}")
    logger.info(f"Logs: {os.path.join(log_dir, 'language_filter.log')}")

    return logger, log_dir


# ═══════════════════════════════════════════════════════════════════════════
# Code Detection Heuristic (gated behind --enable-code-detection)
# ═══════════════════════════════════════════════════════════════════════════

# High-confidence single indicators
_CODE_SINGLE_INDICATORS = [
    "def __init__", "if __name__", "<!DOCTYPE", "public static void main",
    "#!/usr/bin", "#!/bin/bash", "package main", "func main()",
    "#include <", "using namespace", "Console.WriteLine",
    "System.out.println", "import React",
]

# Language keywords (need 2+ matches)
_CODE_KEYWORDS = [
    "def ", "class ", "import ", "from ", "return ", "function ",
    "const ", "let ", "var ", "async ", "await ", "yield ",
    "#include", "SELECT ", "INSERT ", "CREATE TABLE",
    "fn ", "pub fn", "impl ", "struct ", "enum ",
    "package ", "interface ", "throws ",
    "elif ", "else:", "except:", "finally:",
]

# Structural patterns
_CODE_STRUCTURAL = ["=>", "->", "::", "===", "!==", "&&", "||"]


def is_code(text: str) -> bool:
    """
    Heuristic check if text is primarily code.
    Checks first 3000 chars for code fences, keywords, structural patterns.
    """
    sample = text[:3000]

    # Code fences
    if "```" in sample:
        return True

    # Single high-confidence indicators
    for indicator in _CODE_SINGLE_INDICATORS:
        if indicator in sample:
            return True

    # Keyword count (need 2+)
    kw_count = sum(1 for kw in _CODE_KEYWORDS if kw in sample)
    if kw_count >= 2:
        return True

    # Structural pattern count (need 3+)
    struct_count = sum(1 for pat in _CODE_STRUCTURAL if pat in sample)
    if struct_count >= 3:
        return True

    # High brace/paren ratio
    alpha_count = sum(1 for c in sample if c.isalpha())
    if alpha_count > 0:
        brace_count = sum(1 for c in sample if c in "{}()[];")
        if brace_count / alpha_count > 0.08:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════
# Text Extraction
# ═══════════════════════════════════════════════════════════════════════════

def make_output_basename(filepath: str) -> str:
    """
    Generate output basename from grandparent_parent_filename.
    /data/folder1/subfolder/train.jsonl → folder1_subfolder_train
    /data/folder1/train.jsonl → folder1_train
    /train.jsonl → train
    """
    parts = Path(filepath).resolve().parts
    name = Path(filepath).stem  # filename without extension

    if len(parts) >= 4:
        # grandparent + parent + name
        return f"{parts[-3]}_{parts[-2]}_{name}"
    elif len(parts) >= 3:
        # parent + name
        return f"{parts[-2]}_{name}"
    else:
        return name


def extract_text_by_key(row: dict, key: str) -> str:
    """
    Extract text from a JSONL row using the specified key.

    - key='messages' → ChatML: concatenate all content from messages array
    - key='conversations' → ShareGPT: concatenate all value/content from conversations array
    - any other key → treat as flat string field
    - If key not found → return empty string
    """
    val = row.get(key)
    if val is None:
        return ""

    # ChatML messages array
    if key == "messages" and isinstance(val, list):
        parts = []
        for msg in val:
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    parts.append(c.strip())
        return "\n".join(parts)

    # ShareGPT conversations array
    if key == "conversations" and isinstance(val, list):
        parts = []
        for turn in val:
            if isinstance(turn, dict):
                c = turn.get("value") or turn.get("content") or ""
                if isinstance(c, str) and c.strip():
                    parts.append(c.strip())
        return "\n".join(parts)

    # Flat string field
    if isinstance(val, str):
        return val.strip()

    # If it's a list of strings (rare but possible)
    if isinstance(val, list):
        parts = [item for item in val if isinstance(item, str) and item.strip()]
        return "\n".join(parts)

    return ""


# ═══════════════════════════════════════════════════════════════════════════
# Language Detection
# ═══════════════════════════════════════════════════════════════════════════

ALLOWED_LANGS = {"en"}


def detect_language_single(text: str) -> tuple:
    """
    Run fasttext on a single text string.
    Returns (lang_code: str, confidence: float)
    """
    if not text or len(text.strip()) < 10:
        return ("unknown", 0.0)
    try:
        if _fasttext_model is None:
            return ("unknown", 0.0)
        clean = text.replace("\n", " ").replace("\r", " ")
        predictions = _fasttext_model.f.predict(clean, 1, 0.0, "strict")
        if not predictions:
            return ("unknown", 0.0)
        return (predictions[0][1].replace("__label__", ""), predictions[0][0])
    except Exception:
        return ("unknown", 0.0)


def detect_language(text: str, confidence_threshold: float,
                    chunk_size: int = 5000, max_chunks: int = 20,
                    min_en_ratio: float = 0.9) -> tuple:
    """
    Detect language using fasttext with chunked detection for long texts.

    Short texts (≤ chunk_size): single fasttext call.
    Long texts: split into evenly-spaced chunks, run fasttext on each,
    compute English chunk ratio, decide based on min_en_ratio.

    Returns (lang_code: str, confidence: float, decision: str, is_mixed: bool,
             chunk_info: dict or None)
    """
    if not text or len(text.strip()) < 10:
        return ("unknown", 0.0, "reject", False, None)

    # Short text — single call, no chunking needed
    # Only skip chunking for genuinely short texts (< 10K chars)
    # where contamination risk is minimal
    _CHUNK_THRESHOLD = 10000
    if len(text) <= _CHUNK_THRESHOLD:
        lang, conf = detect_language_single(text)
        if lang == "unknown":
            return ("unknown", 0.0, "reject", False, None)
        if lang in ALLOWED_LANGS and conf >= confidence_threshold:
            return (lang, conf, "keep", False, None)
        elif lang in ALLOWED_LANGS and conf < confidence_threshold:
            return (lang, conf, "reject", False, None)
        else:
            return (lang, conf, "reject", False, None)

    # Long text — chunked detection
    text_len = len(text)
    # For texts smaller than chunk_size, create smaller chunks
    effective_chunk_size = min(chunk_size, max(1000, text_len // 4))
    num_chunks = min(max_chunks, max(2, text_len // effective_chunk_size))
    stride = text_len // num_chunks

    chunk_results = []
    lang_counts = defaultdict(int)

    for i in range(num_chunks):
        start = i * stride
        end = min(start + effective_chunk_size, text_len)
        chunk = text[start:end]
        lang, conf = detect_language_single(chunk)

        # Apply confidence threshold per chunk:
        # if confidence is below threshold, treat as "uncertain" not as that language
        if conf < confidence_threshold:
            lang = "uncertain"

        chunk_results.append((lang, conf))
        lang_counts[lang] += 1

    total_chunks = len(chunk_results)
    en_chunks = lang_counts.get("en", 0)
    uncertain_chunks = lang_counts.get("uncertain", 0)
    en_ratio = en_chunks / total_chunks if total_chunks > 0 else 0

    # Determine dominant language (most frequent across chunks)
    dominant_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "unknown"

    # Average confidence of dominant language chunks
    dominant_confs = [conf for lang, conf in chunk_results if lang == dominant_lang]
    avg_conf = sum(dominant_confs) / len(dominant_confs) if dominant_confs else 0.0

    # Mixed language: more than one language detected across chunks
    real_langs = {l for l in lang_counts if l not in ("unknown", "uncertain")}
    is_mixed = len(real_langs) > 1

    chunk_info = {
        "total_chunks": total_chunks,
        "en_chunks": en_chunks,
        "uncertain_chunks": uncertain_chunks,
        "en_ratio": round(en_ratio, 3),
        "lang_distribution": dict(lang_counts),
    }

    # Decision based on en_ratio threshold
    # If all chunks are unknown/uncertain, keep the row (same as short-text unknown behavior)
    if not real_langs:
        return ("unknown", 0.0, "reject", False, chunk_info)

    if en_ratio >= min_en_ratio:
        return ("en", avg_conf, "keep", is_mixed, chunk_info)
    elif dominant_lang in ALLOWED_LANGS:
        # Dominant is English but ratio below threshold — too much contamination
        return ("en", avg_conf, "reject", is_mixed, chunk_info)
    else:
        return (dominant_lang, avg_conf, "reject", is_mixed, chunk_info)


# ═══════════════════════════════════════════════════════════════════════════
# Byte-Range Line-Boundary Alignment
# ═══════════════════════════════════════════════════════════════════════════

def find_line_boundary(filepath: str, offset: int, file_size: int,
                       max_scan_bytes: int = 10 * 1024 * 1024) -> int:
    """
    From `offset`, read forward in 1 MB chunks until a newline is found.
    Returns the byte position immediately after the newline (start of next line).

    Special cases:
      - offset == 0 → returns 0
      - offset >= file_size → returns file_size
      - No newline within max_scan_bytes → returns offset (warns)
    """
    if offset == 0:
        return 0
    if offset >= file_size:
        return file_size

    chunk_size = 1 * 1024 * 1024  # 1 MB
    scanned = 0

    try:
        with open(filepath, "rb") as f:
            f.seek(offset)
            while scanned < max_scan_bytes:
                remaining = file_size - (offset + scanned)
                if remaining <= 0:
                    return file_size
                to_read = min(chunk_size, remaining, max_scan_bytes - scanned)
                chunk = f.read(to_read)
                if not chunk:
                    return file_size
                nl_pos = chunk.find(b"\n")
                if nl_pos >= 0:
                    return offset + scanned + nl_pos + 1
                scanned += len(chunk)
    except Exception as e:
        print(f"[WARN] Line boundary scan failed at offset {offset} in "
              f"{os.path.basename(filepath)}: {e}", file=sys.stderr)
        return offset

    print(f"[WARN] No newline found within {max_scan_bytes // (1024*1024)} MB "
          f"of offset {offset} in {os.path.basename(filepath)}. "
          f"Using raw offset.", file=sys.stderr)
    return offset


def split_file_into_parts(filepath: str, file_size: int,
                          split_threshold_bytes: int) -> list:
    """
    Split a file into byte-range parts with line-boundary alignment.

    Returns list of tuples:
        (filepath, start_byte, end_byte, part_idx, total_parts)
    """
    if file_size <= split_threshold_bytes:
        return [(filepath, 0, file_size, 0, 1)]

    num_parts = max(1, (file_size + split_threshold_bytes - 1) // split_threshold_bytes)
    raw_chunk = file_size // num_parts

    boundaries = [0]
    for i in range(1, num_parts):
        raw_offset = i * raw_chunk
        aligned = find_line_boundary(filepath, raw_offset, file_size)
        # Avoid duplicate boundaries
        if aligned > boundaries[-1]:
            boundaries.append(aligned)
        # If aligned == file_size, no point adding more
        if aligned >= file_size:
            break
    if boundaries[-1] < file_size:
        boundaries.append(file_size)

    total_parts = len(boundaries) - 1
    parts = []
    for idx in range(total_parts):
        parts.append((filepath, boundaries[idx], boundaries[idx + 1], idx, total_parts))

    return parts


# ═══════════════════════════════════════════════════════════════════════════
# Input CSV Parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_file_list(csv_path: str, default_key: str) -> list:
    """
    Parse the input CSV file.
    Returns list of (file_path, key) tuples — one per .jsonl file discovered.

    CSV format:
        path,key
        /data/folder1,messages
        /data/file.jsonl,text
        /data/folder2,
        /data/folder3

    - If path is folder → scan for .jsonl (non-recursive)
    - If path is file → use directly
    - If key is empty/missing → use default_key
    """
    entries = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return entries

        # Normalize header
        header = [h.strip().lower() for h in header]
        path_idx = 0  # assume first column is path
        key_idx = None

        if "path" in header:
            path_idx = header.index("path")
        if "key" in header:
            key_idx = header.index("key")

        for row_num, row in enumerate(reader, start=2):
            if not row or not row[0].strip():
                continue

            path = row[path_idx].strip()
            key = ""
            if key_idx is not None and key_idx < len(row):
                key = row[key_idx].strip()
            if not key:
                key = default_key

            if not key:
                print(f"[ERROR] Row {row_num}: no key specified and no --default-key. "
                      f"Path: {path}", file=sys.stderr)
                sys.exit(1)

            path = os.path.abspath(path)

            if os.path.isfile(path):
                if path.endswith(".jsonl"):
                    entries.append((path, key))
                else:
                    print(f"[WARN] Row {row_num}: {path} is not a .jsonl file, skipping.",
                          file=sys.stderr)
            elif os.path.isdir(path):
                # Non-recursive scan
                found = 0
                for fn in sorted(os.listdir(path)):
                    if fn.endswith(".jsonl"):
                        entries.append((os.path.join(path, fn), key))
                        found += 1
                if found == 0:
                    print(f"[WARN] Row {row_num}: no .jsonl files in {path}",
                          file=sys.stderr)
            else:
                print(f"[WARN] Row {row_num}: {path} does not exist, skipping.",
                      file=sys.stderr)

    return entries


def _validate_file_key(filepath: str, key: str, logger) -> bool:
    """
    Check if the specified key exists in the first 10 and last 10 lines of a JSONL file.
    Returns True if key is found in at least one row.
    """
    found = 0
    checked = 0

    try:
        # Head: first 10 lines
        with open(filepath, "rb") as f:
            for _ in range(10):
                raw = f.readline()
                if not raw:
                    break
                try:
                    row = json_loads(raw.decode("utf-8", errors="replace"))
                    if isinstance(row, dict):
                        checked += 1
                        if key in row:
                            found += 1
                except (ValueError, UnicodeDecodeError):
                    continue

        # Tail: last 10 lines
        with open(filepath, "rb") as f:
            # Seek to end, read backwards to find last 10 lines
            f.seek(0, 2)
            file_size = f.tell()
            # Read last 1MB to find tail lines
            read_size = min(file_size, 1024 * 1024)
            f.seek(file_size - read_size)
            tail_data = f.read(read_size)
            tail_lines = tail_data.split(b"\n")
            # Take last 10 non-empty lines
            tail_lines = [l for l in tail_lines if l.strip()][-10:]

            for raw in tail_lines:
                try:
                    row = json_loads(raw.decode("utf-8", errors="replace"))
                    if isinstance(row, dict):
                        checked += 1
                        if key in row:
                            found += 1
                except (ValueError, UnicodeDecodeError):
                    continue

    except Exception as e:
        logger.warning(f"Key validation failed for {os.path.basename(filepath)}: {e}")
        return True  # On error, don't skip — let processing handle it

    if checked == 0:
        logger.warning(f"Key validation: {os.path.basename(filepath)} — "
                       f"no valid JSON rows found in head/tail")
        return False

    if found == 0:
        logger.warning(f"Key validation: {os.path.basename(filepath)} — "
                       f"key '{key}' not found in any of {checked} sampled rows. "
                       f"Available keys: {_get_sample_keys(filepath)}")
        return False

    logger.info(f"Key validation: {os.path.basename(filepath)} — "
                f"key '{key}' found in {found}/{checked} sampled rows ✓")
    return True


def _get_sample_keys(filepath: str) -> str:
    """Get keys from the first valid JSON row for error reporting."""
    try:
        with open(filepath, "rb") as f:
            for _ in range(5):
                raw = f.readline()
                if not raw:
                    break
                try:
                    row = json_loads(raw.decode("utf-8", errors="replace"))
                    if isinstance(row, dict):
                        return str(list(row.keys())[:10])
                except Exception:
                    continue
    except Exception:
        pass
    return "(could not read)"


# ═══════════════════════════════════════════════════════════════════════════
# Stats Helpers
# ═══════════════════════════════════════════════════════════════════════════

_SIZE_BUCKETS = ["0-1KB", "1-10KB", "10-100KB", "100KB-1MB", "1-10MB", "10MB+"]
_SIZE_THRESHOLDS = [1024, 10240, 102400, 1048576, 10485760]  # 1K, 10K, 100K, 1M, 10M

def size_bucket(nbytes: int) -> str:
    for i, threshold in enumerate(_SIZE_THRESHOLDS):
        if nbytes < threshold:
            return _SIZE_BUCKETS[i]
    return _SIZE_BUCKETS[-1]


_CONF_BUCKETS = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]

def conf_bucket(confidence: float) -> str:
    if confidence < 0.5:
        return "<0.5"
    elif confidence < 0.6:
        return "0.5-0.6"
    elif confidence < 0.7:
        return "0.6-0.7"
    elif confidence < 0.8:
        return "0.7-0.8"
    elif confidence < 0.9:
        return "0.8-0.9"
    else:
        return "0.9-1.0"


def detect_schema(row: dict, key: str) -> str:
    """Detect the schema/format of a JSONL row."""
    if "messages" in row and isinstance(row.get("messages"), list):
        return "chatml"
    if "conversations" in row and isinstance(row.get("conversations"), list):
        return "sharegpt"
    if key in row:
        val = row[key]
        if isinstance(val, str):
            return "flat"
        if isinstance(val, list):
            return "flat"
    return "other"


def new_minmax():
    """Return a fresh min/max/sum/count accumulator."""
    return {"sum": 0.0, "count": 0, "min": float("inf"), "max": 0.0}


def update_minmax(acc: dict, value: float):
    acc["sum"] += value
    acc["count"] += 1
    if value < acc["min"]:
        acc["min"] = value
    if value > acc["max"]:
        acc["max"] = value


def merge_minmax(a: dict, b: dict) -> dict:
    return {
        "sum": a["sum"] + b["sum"],
        "count": a["count"] + b["count"],
        "min": min(a["min"], b["min"]),
        "max": max(a["max"], b["max"]),
    }


def finalize_minmax(acc: dict) -> dict:
    """Add avg, fix inf for empty accumulators."""
    out = dict(acc)
    if out["count"] > 0:
        out["avg"] = round(out["sum"] / out["count"], 2)
    else:
        out["avg"] = 0
        out["min"] = 0
        out["max"] = 0
    out["sum"] = round(out["sum"], 2)
    return out


def merge_bucket_dicts(a: dict, b: dict) -> dict:
    merged = dict(a)
    for k, v in b.items():
        merged[k] = merged.get(k, 0) + v
    return merged


def generate_run_fingerprint(args, file_entries: list) -> str:
    """[14] Generate a reproducibility fingerprint from CLI args + input files."""
    h = hashlib.sha256()
    # CLI args
    for k, v in sorted(vars(args).items()):
        h.update(f"{k}={v}".encode())
    # Input files + sizes
    for fp, key in sorted(file_entries):
        sz = os.path.getsize(fp)
        h.update(f"{fp}:{key}:{sz}".encode())
    return h.hexdigest()[:16]


def _cleanup_stale_shm(logger=None):
    """Remove stale lang_filter_* directories from /dev/shm from previous crashed runs.
    Only removes dirs where the owning process is no longer alive (checked via PID file)."""
    try:
        shm_base = "/dev/shm"
        if not os.path.isdir(shm_base):
            return
        stale = [d for d in os.listdir(shm_base)
                 if d.startswith("lang_filter_") and os.path.isdir(os.path.join(shm_base, d))]
        for d in stale:
            full_path = os.path.join(shm_base, d)
            pid_file = os.path.join(full_path, ".pid")
            try:
                # Check if owning process is still alive
                if os.path.exists(pid_file):
                    with open(pid_file, "r") as f:
                        pid = int(f.read().strip())
                    try:
                        os.kill(pid, 0)  # signal 0 = check if alive
                        # Process is alive — skip this dir
                        if logger:
                            logger.info(f"CLEANUP: Skipping /dev/shm/{d} — "
                                        f"owner PID {pid} is still running")
                        continue
                    except OSError:
                        pass  # Process is dead — safe to remove

                shutil.rmtree(full_path)
                if logger:
                    logger.info(f"CLEANUP: Removed stale /dev/shm/{d}")
            except Exception as e:
                if logger:
                    logger.warning(f"CLEANUP: Could not remove /dev/shm/{d}: {e}")
    except Exception:
        pass


def _write_pid_file(parts_dir: str):
    """Write current PID to parts dir for stale detection."""
    try:
        pid_file = os.path.join(parts_dir, ".pid")
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass


class RAMMonitor:
    """Background thread that monitors RAM usage every N seconds.
    Logs warnings when RAM exceeds threshold."""

    def __init__(self, parts_dir: str, threshold: float = 0.90,
                 interval: int = 30, logger=None):
        self.parts_dir = parts_dir
        self.threshold = threshold
        self.interval = interval
        self.logger = logger
        self._stop_event = threading.Event()
        self._thread = None
        self._warned = False

    def start(self):
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _get_ram_info(self) -> tuple:
        """Returns (used_pct, available_gb, shm_used_gb) or None on failure."""
        try:
            import psutil
            ram = psutil.virtual_memory()
            used_pct = ram.percent / 100.0
            available_gb = ram.available / (1024**3)
        except ImportError:
            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            meminfo[parts[0].rstrip(":")] = int(parts[1]) * 1024
                total = meminfo.get("MemTotal", 1)
                available = meminfo.get("MemAvailable", 0)
                used_pct = 1.0 - (available / total) if total > 0 else 0
                available_gb = available / (1024**3)
            except Exception:
                return None

        # Check /dev/shm usage
        shm_used_gb = 0
        try:
            if os.path.isdir(self.parts_dir):
                total_size = 0
                for root, dirs, files in os.walk(self.parts_dir):
                    for f in files:
                        total_size += os.path.getsize(os.path.join(root, f))
                shm_used_gb = total_size / (1024**3)
        except Exception:
            pass

        return (used_pct, available_gb, shm_used_gb)

    def _monitor(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(self.interval)
            if self._stop_event.is_set():
                break

            info = self._get_ram_info()
            if info is None:
                continue

            used_pct, available_gb, shm_used_gb = info

            if used_pct >= 0.95:
                if self.logger:
                    self.logger.error(
                        f"RAM CRITICAL: {used_pct*100:.0f}% used, "
                        f"{available_gb:.1f} GB available, "
                        f"parts in /dev/shm: {shm_used_gb:.1f} GB — "
                        f"workers may fail with disk full errors")
            elif used_pct >= self.threshold:
                if self.logger and not self._warned:
                    self.logger.warning(
                        f"RAM WARNING: {used_pct*100:.0f}% used, "
                        f"{available_gb:.1f} GB available, "
                        f"parts in /dev/shm: {shm_used_gb:.1f} GB")
                    self._warned = True
            else:
                self._warned = False


def check_resume(output_dir: str, output_base: str, file_entries: list, args, logger) -> set:
    """
    Check previous run_* subdirs in output_base for already-processed files.
    Finds the latest manifest.json, verifies config match, and skips files
    that are already processed. Copies their output to current run dir.
    """
    # Find latest manifest across all run_* dirs
    manifest_path = None
    prev_run_dir = None

    if os.path.isdir(output_base):
        for d in sorted(os.listdir(output_base), reverse=True):
            if d.startswith("run_"):
                candidate = os.path.join(output_base, d, "manifest.json")
                if os.path.exists(candidate):
                    manifest_path = candidate
                    prev_run_dir = os.path.join(output_base, d)
                    break  # sorted reverse = latest first

    if not manifest_path:
        logger.info("RESUME: No previous manifest found in any run — processing all files")
        return set()

    logger.info(f"RESUME: Using manifest from {prev_run_dir}")

    try:
        with open(manifest_path, "r") as f:
            prev_manifest = json.load(f)
    except Exception as e:
        logger.warning(f"RESUME: Could not read previous manifest: {e} — processing all files")
        return set()

    # Check if config matches
    prev_config = prev_manifest.get("run_info", {}).get("config", {})
    config_keys = ["confidence", "min_en_ratio", "chunk_size", "max_chunks"]
    config_changed = []
    for k in config_keys:
        prev_val = prev_config.get(k)
        curr_val = getattr(args, k.replace("-", "_"), None)
        if prev_val is not None and curr_val is not None and prev_val != curr_val:
            config_changed.append(f"{k}: {prev_val} → {curr_val}")

    if config_changed:
        logger.warning(f"RESUME: Config changed from previous run — reprocessing all files")
        for change in config_changed:
            logger.warning(f"  {change}")
        return set()

    prev_files = {}
    for pf in prev_manifest.get("per_file", []):
        prev_files[pf["filepath"]] = pf

    skip = set()
    for filepath, key in file_entries:
        if filepath not in prev_files:
            continue

        prev = prev_files[filepath]
        basename = prev.get("basename", "")

        # Check input file hasn't changed
        current_size = os.path.getsize(filepath)
        prev_size = prev.get("input_size_bytes", -1)
        if current_size != prev_size:
            logger.info(f"RESUME: {os.path.basename(filepath)} — size changed "
                        f"({prev_size} → {current_size}), reprocessing")
            continue

        # Check output files exist in previous run dir
        en_path = os.path.join(prev_run_dir, f"{basename}_en.jsonl")
        non_path = os.path.join(prev_run_dir, f"{basename}_non_selected.jsonl")
        unk_path = os.path.join(prev_run_dir, f"{basename}_unknown.jsonl")
        if not os.path.exists(en_path) or not os.path.exists(non_path) or not os.path.exists(unk_path):
            logger.info(f"RESUME: {os.path.basename(filepath)} — output files missing in "
                        f"{prev_run_dir}, reprocessing")
            continue

        # Check row counts match
        expected_kept = prev.get("kept_rows", -1)
        expected_rejected = prev.get("rejected_rows", -1)
        expected_unknown = prev.get("unknown_rows", -1)
        actual_kept = count_file_lines(en_path)
        actual_rejected = count_file_lines(non_path)
        actual_unknown = count_file_lines(unk_path)
        if (actual_kept != expected_kept or actual_rejected != expected_rejected
                or actual_unknown != expected_unknown):
            logger.info(f"RESUME: {os.path.basename(filepath)} — output line counts don't match "
                        f"manifest, reprocessing")
            continue

        # Copy previous output files to current run dir
        for src in [en_path, non_path, unk_path]:
            dst = os.path.join(output_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            logger.info(f"RESUME: Copied {os.path.basename(src)} from previous run")

        skip.add(filepath)
        logger.info(f"RESUME: Skipping {os.path.basename(filepath)} — already processed "
                    f"(kept={expected_kept}, rejected={expected_rejected}, unknown={expected_unknown})")

    if skip:
        logger.info(f"RESUME: Skipping {len(skip)} file(s), processing {len(file_entries) - len(skip)} file(s)")

    return skip


def select_parts_dir(output_dir: str, total_input_bytes: int, run_timestamp: str,
                     ram_threshold: float = 0.90, logger=None) -> tuple:
    """
    Select parts directory: RAM (/dev/shm) if available and RAM < threshold,
    otherwise fall back to disk (output_dir/parts/).

    Returns (parts_dir: str, is_ram: bool)
    """
    # Clean up stale lang_filter dirs from previous crashed runs
    _cleanup_stale_shm(logger)

    # Estimate parts space needed: ~1.1x input (en + non_selected parts)
    estimated_parts_bytes = int(total_input_bytes * 1.1)

    # Try /dev/shm (RAM-backed tmpfs)
    shm_path = f"/dev/shm/lang_filter_{run_timestamp}"
    try:
        import psutil
        ram = psutil.virtual_memory()
        ram_used_pct = ram.percent / 100.0
        ram_available = ram.available

        if ram_used_pct < ram_threshold and ram_available > estimated_parts_bytes:
            os.makedirs(shm_path, exist_ok=True)
            _write_pid_file(shm_path)
            if logger:
                logger.info(f"PARTS DIR: Using RAM (/dev/shm) — "
                            f"RAM {ram_used_pct*100:.0f}% used, "
                            f"{ram_available / (1024**3):.1f} GB available, "
                            f"~{estimated_parts_bytes / (1024**3):.1f} GB needed for parts")
            return shm_path, True
        else:
            if logger:
                logger.info(f"PARTS DIR: Using disk (RAM {ram_used_pct*100:.0f}% used, "
                            f"available {ram_available / (1024**3):.1f} GB, "
                            f"need {estimated_parts_bytes / (1024**3):.1f} GB) — "
                            f"falling back to {output_dir}/parts/")
    except ImportError:
        # psutil not installed — try /dev/shm with statvfs
        try:
            stat = os.statvfs("/dev/shm")
            shm_available = stat.f_bavail * stat.f_frsize
            if shm_available > estimated_parts_bytes:
                os.makedirs(shm_path, exist_ok=True)
                _write_pid_file(shm_path)
                if logger:
                    logger.info(f"PARTS DIR: Using RAM (/dev/shm) — "
                                f"{shm_available / (1024**3):.1f} GB available, "
                                f"~{estimated_parts_bytes / (1024**3):.1f} GB needed")
                return shm_path, True
            else:
                if logger:
                    logger.info(f"PARTS DIR: /dev/shm too small "
                                f"({shm_available / (1024**3):.1f} GB available, "
                                f"need {estimated_parts_bytes / (1024**3):.1f} GB) — "
                                f"falling back to disk")
        except Exception:
            pass
    except Exception:
        pass

    # Fallback to disk
    disk_parts = os.path.join(output_dir, "parts")
    os.makedirs(disk_parts, exist_ok=True)
    if logger:
        logger.info(f"PARTS DIR: Using disk — {disk_parts}")
    return disk_parts, False


# ═══════════════════════════════════════════════════════════════════════════
# Worker Initializer
# ═══════════════════════════════════════════════════════════════════════════

# Global for fasttext model (loaded per-worker)
_fasttext_model = None
_fasttext_model_path = None


def worker_init(model_path: str = None):
    """Per-worker init: load fasttext model."""
    global _fasttext_model, _fasttext_model_path
    _fasttext_model_path = model_path
    _fasttext_model = None
    if model_path:
        try:
            import fasttext
            fasttext.FastText.eprint = lambda x: None  # suppress warnings
            _fasttext_model = fasttext.load_model(model_path)
        except Exception as e:
            # Store error message — process_part will check and raise
            _fasttext_model = None
            print(f"[CRITICAL] Worker {os.getpid()} failed to load fasttext model: {e}",
                  file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════
# Worker: Process One Part
# ═══════════════════════════════════════════════════════════════════════════

def process_part_safe(task: dict) -> dict:
    """Wrapper around process_part that catches exceptions and returns error info."""
    try:
        return process_part(task)
    except Exception as e:
        import traceback
        return {
            "_error": True,
            "_error_msg": str(e),
            "_error_traceback": traceback.format_exc(),
            "filepath": task.get("filepath", "unknown"),
            "filename": os.path.basename(task.get("filepath", "unknown")),
            "part_idx": task.get("part_idx", -1),
            "total_parts": task.get("total_parts", 0),
        }


def process_part(task: dict) -> dict:
    """
    Process a single byte-range part of a file.

    task keys:
        filepath, start_byte, end_byte, part_idx, total_parts,
        key, output_dir, confidence, enable_code_detection

    Returns stats dict.
    """
    filepath = task["filepath"]
    start_byte = task["start_byte"]
    end_byte = task["end_byte"]
    part_idx = task["part_idx"]
    total_parts = task["total_parts"]
    key = task["key"]
    output_dir = task["output_dir"]
    parts_dir = task["parts_dir"]
    confidence = task["confidence"]
    chunk_size = task["chunk_size"]
    max_chunks = task["max_chunks"]
    min_en_ratio = task["min_en_ratio"]
    code_detect = task["enable_code_detection"]

    basename = make_output_basename(filepath)

    # [FIX #1] Fail loudly if fasttext model not loaded
    if _fasttext_model is None:
        raise RuntimeError(
            f"Worker {os.getpid()}: fasttext model not loaded. "
            f"Cannot process {os.path.basename(filepath)} part {part_idx}. "
            f"Check model path: {_fasttext_model_path}"
        )

    # Part-indexed output files
    if total_parts > 1:
        suffix = f"_part{part_idx}of{total_parts}"
    else:
        suffix = ""

    en_hi_path = os.path.join(parts_dir, f"{basename}{suffix}_en.jsonl")
    non_sel_path = os.path.join(parts_dir, f"{basename}{suffix}_non_selected.jsonl")
    unknown_path = os.path.join(parts_dir, f"{basename}{suffix}_unknown.jsonl")

    os.makedirs(parts_dir, exist_ok=True)

    stats = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "basename": basename,
        "part_idx": part_idx,
        "total_parts": total_parts,
        "start_byte": start_byte,
        "end_byte": end_byte,
        # Core counts
        "total_rows": 0,
        "kept_rows": 0,
        "rejected_rows": 0,
        "unknown_rows": 0,
        "code_rows": 0,
        "json_errors": 0,
        "key_missing": 0,
        "empty_lines": 0,
        "content_empty": 0,          # [7] key exists but text empty
        "mixed_lang_rows": 0,        # [5] mixed language detection
        "contaminated_rows": 0,      # rows rejected due to low en_ratio
        # Language breakdown
        "lang_counts": defaultdict(int),
        # [1] Word counts
        "kept_word_count": 0,
        "rejected_word_count": 0,
        # [2] Row size distribution
        "kept_row_size": new_minmax(),
        "rejected_row_size": new_minmax(),
        "row_size_buckets_kept": {b: 0 for b in _SIZE_BUCKETS},
        "row_size_buckets_rejected": {b: 0 for b in _SIZE_BUCKETS},
        # [3] Confidence distribution
        "conf_kept": new_minmax(),
        "conf_rejected": new_minmax(),
        "conf_buckets_kept": {b: 0 for b in _CONF_BUCKETS},
        "conf_buckets_rejected": {b: 0 for b in _CONF_BUCKETS},
        # [8] Schema distribution
        "schema_counts": defaultdict(int),
        # [12] Estimated tokens (word_count * 1.3)
        "kept_est_tokens": 0,
        "rejected_est_tokens": 0,
        # [7] Chunked detection stats
        "chunked_rows": 0,
        "single_call_rows": 0,
        "en_ratio_buckets": {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)},
        # Output paths
        "en_path": en_hi_path,
        "non_sel_path": non_sel_path,
        "unknown_path": unknown_path,
        # [11] Timing (set after processing)
        "processing_time_sec": 0.0,
    }

    # Graceful handling: empty byte range
    if start_byte >= end_byte:
        # Write empty files
        open(en_hi_path, "w").close()
        open(non_sel_path, "w").close()
        open(unknown_path, "w").close()
        return stats

    part_t0 = time.time()
    en_hi_fh = open(en_hi_path, "wb")
    non_sel_fh = open(non_sel_path, "wb")
    unknown_fh = open(unknown_path, "wb")

    try:
        with open(filepath, "rb") as f:
            # Seek to start
            f.seek(start_byte)

            # If not at beginning, skip partial line
            if start_byte > 0:
                f.seek(start_byte - 1)
                prev_byte = f.read(1)
                # f is now at start_byte
                if prev_byte != b"\n":
                    f.readline()  # skip partial line

            while True:
                pos = f.tell()
                if pos >= end_byte:
                    break

                raw_line = f.readline()
                if not raw_line:
                    break

                # Normalize: ensure raw line ends with \n for consistent output
                raw_line_out = raw_line if raw_line.endswith(b"\n") else raw_line + b"\n"

                # Track raw byte size before decoding
                line_bytes = len(raw_line)

                # Decode and strip
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
                if not line:
                    stats["empty_lines"] += 1
                    continue

                stats["total_rows"] += 1

                # Parse JSON
                try:
                    row = json_loads(line)
                except (ValueError, json.JSONDecodeError):
                    stats["json_errors"] += 1
                    non_sel_fh.write(raw_line_out)
                    stats["rejected_rows"] += 1
                    update_minmax(stats["rejected_row_size"], line_bytes)
                    stats["row_size_buckets_rejected"][size_bucket(line_bytes)] += 1
                    continue

                if not isinstance(row, dict):
                    stats["json_errors"] += 1
                    non_sel_fh.write(raw_line_out)
                    stats["rejected_rows"] += 1
                    update_minmax(stats["rejected_row_size"], line_bytes)
                    stats["row_size_buckets_rejected"][size_bucket(line_bytes)] += 1
                    continue

                # [8] Schema detection
                schema = detect_schema(row, key)
                stats["schema_counts"][schema] += 1

                # Extract text using specified key
                text = extract_text_by_key(row, key)

                # [7] Content emptiness: key exists but text is empty
                if key in row and not text:
                    stats["content_empty"] += 1

                if not text:
                    stats["key_missing"] += 1
                    unknown_fh.write(raw_line_out)
                    stats["unknown_rows"] += 1
                    stats["lang_counts"]["no_text"] += 1
                    update_minmax(stats["kept_row_size"], line_bytes)
                    stats["row_size_buckets_kept"][size_bucket(line_bytes)] += 1
                    continue

                # Word count for this row
                word_count = text.count(" ") + 1
                est_tokens = int(word_count * 1.3)

                # Code detection (if enabled)
                if code_detect and is_code(text):
                    stats["code_rows"] += 1
                    en_hi_fh.write(raw_line_out)
                    stats["kept_rows"] += 1
                    stats["lang_counts"]["code"] += 1
                    stats["kept_word_count"] += word_count
                    stats["kept_est_tokens"] += est_tokens
                    update_minmax(stats["kept_row_size"], line_bytes)
                    stats["row_size_buckets_kept"][size_bucket(line_bytes)] += 1
                    continue

                # Detect language (chunked for long texts)
                lang, conf, decision, is_mixed, chunk_info = detect_language(
                    text, confidence, chunk_size, max_chunks, min_en_ratio)
                stats["lang_counts"][lang] += 1

                # [7] Track chunked vs single-call detection
                if chunk_info is not None:
                    stats["chunked_rows"] += 1
                    # Bucket the en_ratio
                    er = chunk_info["en_ratio"]
                    bucket_idx = min(9, int(er * 10))
                    bucket_key = f"{bucket_idx/10:.1f}-{(bucket_idx+1)/10:.1f}"
                    stats["en_ratio_buckets"][bucket_key] += 1
                else:
                    stats["single_call_rows"] += 1

                # [5] Mixed language tracking
                if is_mixed:
                    stats["mixed_lang_rows"] += 1

                # Track contamination (rejected due to en_ratio below threshold)
                if chunk_info and decision == "reject" and lang in ALLOWED_LANGS:
                    stats["contaminated_rows"] += 1

                if decision == "keep":
                    if lang == "unknown":
                        # Unknown goes to separate file
                        unknown_fh.write(raw_line_out)
                        stats["unknown_rows"] += 1
                    else:
                        en_hi_fh.write(raw_line_out)
                        stats["kept_rows"] += 1
                        stats["kept_word_count"] += word_count
                        stats["kept_est_tokens"] += est_tokens
                        update_minmax(stats["kept_row_size"], line_bytes)
                        stats["row_size_buckets_kept"][size_bucket(line_bytes)] += 1
                        if conf > 0:
                            update_minmax(stats["conf_kept"], conf)
                            stats["conf_buckets_kept"][conf_bucket(conf)] += 1
                elif decision == "reject":
                    non_sel_fh.write(raw_line_out)
                    stats["rejected_rows"] += 1
                    stats["rejected_word_count"] += word_count
                    stats["rejected_est_tokens"] += est_tokens
                    update_minmax(stats["rejected_row_size"], line_bytes)
                    stats["row_size_buckets_rejected"][size_bucket(line_bytes)] += 1
                    if conf > 0:
                        update_minmax(stats["conf_rejected"], conf)
                        stats["conf_buckets_rejected"][conf_bucket(conf)] += 1
                else:
                    # Safety net — unknown/uncertain fallback
                    unknown_fh.write(raw_line_out)
                    stats["unknown_rows"] += 1

    finally:
        en_hi_fh.close()
        non_sel_fh.close()
        unknown_fh.close()

    stats["processing_time_sec"] = round(time.time() - part_t0, 2)

    # Convert defaultdicts to regular dicts for serialization
    stats["lang_counts"] = dict(stats["lang_counts"])
    stats["schema_counts"] = dict(stats["schema_counts"])

    # [V1] Row accounting assertion
    expected = stats["total_rows"]
    actual = stats["kept_rows"] + stats["rejected_rows"] + stats["unknown_rows"]
    if expected != actual:
        stats["accounting_error"] = (f"kept({stats['kept_rows']}) + "
                                     f"rejected({stats['rejected_rows']}) + "
                                     f"unknown({stats['unknown_rows']}) = {actual} "
                                     f"!= total({expected})")
    else:
        stats["accounting_error"] = None

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Merge Part Files
# ═══════════════════════════════════════════════════════════════════════════

def merge_part_files(part_files: list, merged_path: str):
    """Concatenate part files into a single merged file (binary-safe)."""
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, "wb") as out:
        for pf in part_files:
            if not os.path.exists(pf):
                continue
            with open(pf, "rb") as inp:
                while True:
                    chunk = inp.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    out.write(chunk)


# ═══════════════════════════════════════════════════════════════════════════
# Stats Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_stats(all_stats: list, file_entries: list) -> tuple:
    """
    Group stats by source file, merge part stats.
    Returns (per_file_stats: list[dict], global_stats: dict)
    """
    by_file = defaultdict(list)
    for s in all_stats:
        by_file[s["filepath"]].append(s)

    # Build input size + mtime map
    file_info = {}
    for fp, key in file_entries:
        file_info[fp] = {
            "size_bytes": os.path.getsize(fp),
            "mtime": os.path.getmtime(fp),
            "key": key,
        }

    _SIMPLE_SUM_KEYS = [
        "total_rows", "kept_rows", "rejected_rows", "unknown_rows",
        "code_rows", "json_errors", "key_missing", "empty_lines",
        "content_empty", "mixed_lang_rows", "contaminated_rows",
        "kept_word_count", "rejected_word_count",
        "kept_est_tokens", "rejected_est_tokens",
        "chunked_rows", "single_call_rows",
    ]

    per_file = []
    global_totals = {k: 0 for k in _SIMPLE_SUM_KEYS}
    global_totals.update({
        "total_files": 0,
        "total_input_bytes": 0,
        "lang_counts": defaultdict(int),
        "schema_counts": defaultdict(int),
        "kept_row_size": new_minmax(),
        "rejected_row_size": new_minmax(),
        "row_size_buckets_kept": {b: 0 for b in _SIZE_BUCKETS},
        "row_size_buckets_rejected": {b: 0 for b in _SIZE_BUCKETS},
        "conf_kept": new_minmax(),
        "conf_rejected": new_minmax(),
        "conf_buckets_kept": {b: 0 for b in _CONF_BUCKETS},
        "conf_buckets_rejected": {b: 0 for b in _CONF_BUCKETS},
        "en_ratio_buckets": {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)},
        "total_processing_time_sec": 0.0,
    })

    for filepath in sorted(by_file.keys()):
        parts = sorted(by_file[filepath], key=lambda s: s["part_idx"])
        info = file_info.get(filepath, {"size_bytes": 0, "mtime": 0, "key": ""})

        merged = {k: 0 for k in _SIMPLE_SUM_KEYS}
        merged.update({
            "filepath": filepath,
            "filename": parts[0]["filename"],
            "basename": parts[0]["basename"],
            "total_parts": parts[0]["total_parts"],
            "input_size_bytes": info["size_bytes"],
            "input_mtime": info["mtime"],
            "key": info["key"],
            "lang_counts": defaultdict(int),
            "schema_counts": defaultdict(int),
            "kept_row_size": new_minmax(),
            "rejected_row_size": new_minmax(),
            "row_size_buckets_kept": {b: 0 for b in _SIZE_BUCKETS},
            "row_size_buckets_rejected": {b: 0 for b in _SIZE_BUCKETS},
            "conf_kept": new_minmax(),
            "conf_rejected": new_minmax(),
            "conf_buckets_kept": {b: 0 for b in _CONF_BUCKETS},
            "conf_buckets_rejected": {b: 0 for b in _CONF_BUCKETS},
            "en_ratio_buckets": {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)},
            "processing_time_sec": 0.0,
        })

        for p in parts:
            for k in _SIMPLE_SUM_KEYS:
                merged[k] += p[k]
            for lang, cnt in p["lang_counts"].items():
                merged["lang_counts"][lang] += cnt
            for schema, cnt in p["schema_counts"].items():
                merged["schema_counts"][schema] += cnt
            merged["kept_row_size"] = merge_minmax(merged["kept_row_size"], p["kept_row_size"])
            merged["rejected_row_size"] = merge_minmax(merged["rejected_row_size"], p["rejected_row_size"])
            merged["row_size_buckets_kept"] = merge_bucket_dicts(merged["row_size_buckets_kept"], p["row_size_buckets_kept"])
            merged["row_size_buckets_rejected"] = merge_bucket_dicts(merged["row_size_buckets_rejected"], p["row_size_buckets_rejected"])
            merged["conf_kept"] = merge_minmax(merged["conf_kept"], p["conf_kept"])
            merged["conf_rejected"] = merge_minmax(merged["conf_rejected"], p["conf_rejected"])
            merged["conf_buckets_kept"] = merge_bucket_dicts(merged["conf_buckets_kept"], p["conf_buckets_kept"])
            merged["conf_buckets_rejected"] = merge_bucket_dicts(merged["conf_buckets_rejected"], p["conf_buckets_rejected"])
            merged["en_ratio_buckets"] = merge_bucket_dicts(merged["en_ratio_buckets"], p["en_ratio_buckets"])
            merged["processing_time_sec"] += p["processing_time_sec"]

        # Accumulate global BEFORE finalizing per-file (raw accumulators)
        global_totals["total_files"] += 1
        global_totals["total_input_bytes"] += merged["input_size_bytes"]
        for k in _SIMPLE_SUM_KEYS:
            global_totals[k] += merged[k]
        for lang, cnt in merged["lang_counts"].items():
            global_totals["lang_counts"][lang] += cnt
        for schema, cnt in merged["schema_counts"].items():
            global_totals["schema_counts"][schema] += cnt
        global_totals["kept_row_size"] = merge_minmax(global_totals["kept_row_size"], merged["kept_row_size"])
        global_totals["rejected_row_size"] = merge_minmax(global_totals["rejected_row_size"], merged["rejected_row_size"])
        global_totals["row_size_buckets_kept"] = merge_bucket_dicts(global_totals["row_size_buckets_kept"], merged["row_size_buckets_kept"])
        global_totals["row_size_buckets_rejected"] = merge_bucket_dicts(global_totals["row_size_buckets_rejected"], merged["row_size_buckets_rejected"])
        global_totals["conf_kept"] = merge_minmax(global_totals["conf_kept"], merged["conf_kept"])
        global_totals["conf_rejected"] = merge_minmax(global_totals["conf_rejected"], merged["conf_rejected"])
        global_totals["conf_buckets_kept"] = merge_bucket_dicts(global_totals["conf_buckets_kept"], merged["conf_buckets_kept"])
        global_totals["conf_buckets_rejected"] = merge_bucket_dicts(global_totals["conf_buckets_rejected"], merged["conf_buckets_rejected"])
        global_totals["en_ratio_buckets"] = merge_bucket_dicts(global_totals["en_ratio_buckets"], merged["en_ratio_buckets"])
        global_totals["total_processing_time_sec"] += merged["processing_time_sec"]

        # NOW finalize per-file (after global merge used raw values)
        merged["lang_counts"] = dict(merged["lang_counts"])
        merged["schema_counts"] = dict(merged["schema_counts"])
        merged["kept_row_size"] = finalize_minmax(merged["kept_row_size"])
        merged["rejected_row_size"] = finalize_minmax(merged["rejected_row_size"])
        merged["conf_kept"] = finalize_minmax(merged["conf_kept"])
        merged["conf_rejected"] = finalize_minmax(merged["conf_rejected"])
        merged["processing_time_sec"] = round(merged["processing_time_sec"], 2)

        merged["kept_pct"] = (
            f"{merged['kept_rows'] / merged['total_rows'] * 100:.1f}%"
            if merged["total_rows"] > 0 else "N/A"
        )
        # [10] Per-source quality score (rejection rate)
        merged["rejection_rate"] = (
            round(merged["rejected_rows"] / merged["total_rows"] * 100, 1)
            if merged["total_rows"] > 0 else 0
        )
        # [11] Throughput
        merged["rows_per_sec"] = (
            round(merged["total_rows"] / merged["processing_time_sec"], 1)
            if merged["processing_time_sec"] > 0 else 0
        )
        merged["mb_per_sec"] = (
            round(merged["input_size_bytes"] / (1024 * 1024) / merged["processing_time_sec"], 1)
            if merged["processing_time_sec"] > 0 else 0
        )

        per_file.append(merged)

    # Finalize global
    global_totals["lang_counts"] = dict(global_totals["lang_counts"])
    global_totals["schema_counts"] = dict(global_totals["schema_counts"])
    global_totals["kept_row_size"] = finalize_minmax(global_totals["kept_row_size"])
    global_totals["rejected_row_size"] = finalize_minmax(global_totals["rejected_row_size"])
    global_totals["conf_kept"] = finalize_minmax(global_totals["conf_kept"])
    global_totals["conf_rejected"] = finalize_minmax(global_totals["conf_rejected"])
    global_totals["total_processing_time_sec"] = round(global_totals["total_processing_time_sec"], 2)

    global_totals["kept_pct"] = (
        f"{global_totals['kept_rows'] / global_totals['total_rows'] * 100:.1f}%"
        if global_totals["total_rows"] > 0 else "N/A"
    )
    # [9] Data retention ratio
    global_totals["retention_ratio_rows"] = (
        round(global_totals["kept_rows"] / global_totals["total_rows"] * 100, 1)
        if global_totals["total_rows"] > 0 else 0
    )
    global_totals["retention_ratio_words"] = (
        round(global_totals["kept_word_count"] / (global_totals["kept_word_count"] + global_totals["rejected_word_count"]) * 100, 1)
        if (global_totals["kept_word_count"] + global_totals["rejected_word_count"]) > 0 else 0
    )
    global_totals["retention_ratio_est_tokens"] = (
        round(global_totals["kept_est_tokens"] / (global_totals["kept_est_tokens"] + global_totals["rejected_est_tokens"]) * 100, 1)
        if (global_totals["kept_est_tokens"] + global_totals["rejected_est_tokens"]) > 0 else 0
    )
    # [4] Top rejected languages (sorted)
    rejected_langs = {k: v for k, v in global_totals["lang_counts"].items()
                      if k not in ALLOWED_LANGS and k not in ("unknown", "no_text", "code")}
    global_totals["top_rejected_langs"] = dict(
        sorted(rejected_langs.items(), key=lambda x: x[1], reverse=True)
    )

    # [10] Per-source quality ranking (worst first)
    per_file.sort(key=lambda f: f["rejection_rate"], reverse=True)

    return per_file, global_totals


def write_stats_csv(per_file: list, global_totals: dict, csv_path: str):
    """Write per-file stats + global totals row to CSV."""
    all_langs = set()
    for pf in per_file:
        all_langs.update(pf["lang_counts"].keys())
    all_langs.update(global_totals["lang_counts"].keys())
    lang_cols = sorted(all_langs)

    all_schemas = set()
    for pf in per_file:
        all_schemas.update(pf["schema_counts"].keys())
    schema_cols = sorted(all_schemas)

    fieldnames = [
        "filename", "input_size_mb", "total_parts", "key",
        "total_rows", "kept_rows", "rejected_rows", "kept_pct", "rejection_rate",
        "unknown_rows", "code_rows", "mixed_lang_rows", "contaminated_rows",
        "json_errors", "key_missing", "content_empty", "empty_lines",
        "kept_word_count", "rejected_word_count",
        "kept_est_tokens", "rejected_est_tokens",
        "kept_row_size_avg", "kept_row_size_min", "kept_row_size_max",
        "rejected_row_size_avg", "rejected_row_size_min", "rejected_row_size_max",
        "conf_kept_avg", "conf_kept_min", "conf_kept_max",
        "conf_rejected_avg",
        "processing_time_sec", "rows_per_sec", "mb_per_sec",
    ] + [f"schema_{s}" for s in schema_cols
    ] + [f"lang_{l}" for l in lang_cols]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    def _build_row(data, is_global=False):
        row = {
            "filename": data.get("filename", "=== TOTAL ==="),
            "input_size_mb": round(data.get("input_size_bytes", data.get("total_input_bytes", 0)) / (1024 * 1024), 2),
            "total_parts": data.get("total_parts", ""),
            "key": data.get("key", ""),
            "total_rows": data["total_rows"],
            "kept_rows": data["kept_rows"],
            "rejected_rows": data["rejected_rows"],
            "kept_pct": data["kept_pct"],
            "rejection_rate": data.get("rejection_rate", ""),
            "unknown_rows": data["unknown_rows"],
            "code_rows": data["code_rows"],
            "mixed_lang_rows": data["mixed_lang_rows"],
            "contaminated_rows": data["contaminated_rows"],
            "json_errors": data["json_errors"],
            "key_missing": data["key_missing"],
            "content_empty": data["content_empty"],
            "empty_lines": data["empty_lines"],
            "kept_word_count": data["kept_word_count"],
            "rejected_word_count": data["rejected_word_count"],
            "kept_est_tokens": data["kept_est_tokens"],
            "rejected_est_tokens": data["rejected_est_tokens"],
            "kept_row_size_avg": data["kept_row_size"].get("avg", 0),
            "kept_row_size_min": data["kept_row_size"].get("min", 0),
            "kept_row_size_max": data["kept_row_size"].get("max", 0),
            "rejected_row_size_avg": data["rejected_row_size"].get("avg", 0),
            "rejected_row_size_min": data["rejected_row_size"].get("min", 0),
            "rejected_row_size_max": data["rejected_row_size"].get("max", 0),
            "conf_kept_avg": data["conf_kept"].get("avg", 0),
            "conf_kept_min": data["conf_kept"].get("min", 0),
            "conf_kept_max": data["conf_kept"].get("max", 0),
            "conf_rejected_avg": data["conf_rejected"].get("avg", 0),
            "processing_time_sec": data.get("processing_time_sec",
                                            data.get("total_processing_time_sec", 0)),
            "rows_per_sec": data.get("rows_per_sec", ""),
            "mb_per_sec": data.get("mb_per_sec", ""),
        }
        for s in schema_cols:
            row[f"schema_{s}"] = data.get("schema_counts", {}).get(s, 0)
        for l in lang_cols:
            row[f"lang_{l}"] = data.get("lang_counts", {}).get(l, 0)
        return row

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pf in per_file:
            writer.writerow(_build_row(pf))
        writer.writerow(_build_row(global_totals, is_global=True))


def write_manifest_json(per_file: list, global_totals: dict, run_info: dict,
                        json_path: str):
    """Write comprehensive manifest JSON with full detail."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # [13] Provenance: map output files to input files
    provenance = {}
    for pf in per_file:
        provenance[pf["basename"]] = {
            "input_file": pf["filepath"],
            "input_size_bytes": pf["input_size_bytes"],
            "input_mtime": pf["input_mtime"],
            "input_mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                              time.localtime(pf["input_mtime"])),
            "output_en": f"{pf['basename']}_en.jsonl",
            "output_non_selected": f"{pf['basename']}_non_selected.jsonl",
            "output_unknown": f"{pf['basename']}_unknown.jsonl",
            "key": pf["key"],
        }

    data = {
        "run_info": run_info,
        "global_summary": global_totals,
        "per_file": per_file,
        "provenance": provenance,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def preflight_fasttext_check(model_path: str, logger):
    """[V11] Fail fast if fasttext is not installed or model is missing."""
    try:
        import fasttext
    except ImportError:
        logger.error("PRE-FLIGHT FAILED: fasttext is not installed. "
                      "Run: pip install fasttext-wheel --break-system-packages")
        sys.exit(1)

    if not os.path.isfile(model_path):
        logger.error(f"PRE-FLIGHT FAILED: fasttext model not found at {model_path}. "
                      "Download: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz")
        sys.exit(1)

    # Smoke test
    try:
        fasttext.FastText.eprint = lambda x: None  # suppress warnings
        model = fasttext.load_model(model_path)
        preds = model.f.predict("This is a simple English test sentence for validation.", 1, 0.0, "strict")
        if preds and preds[0][1] != "__label__en":
            logger.warning(f"fasttext smoke test returned '{preds[0][1]}' instead of 'en' — "
                           f"results may be unreliable")
        logger.info(f"PRE-FLIGHT: fasttext model OK ({model_path})")
        del model
    except Exception as e:
        logger.error(f"PRE-FLIGHT FAILED: Could not load fasttext model: {e}")
        sys.exit(1)


def preflight_disk_space_check(output_dir: str, total_input_bytes: int, logger):
    """[V9] Check if output directory has enough disk space.
    Disk needed: ~1x (en + non_selected per file) + logs/manifest overhead."""
    required_bytes = int(total_input_bytes * 1.1)  # 1.1x safety margin
    try:
        stat = os.statvfs(output_dir)
        available_bytes = stat.f_bavail * stat.f_frsize
        required_gb = required_bytes / (1024**3)
        available_gb = available_bytes / (1024**3)
        if available_bytes < required_bytes:
            logger.error(f"PRE-FLIGHT FAILED: Insufficient disk space in {output_dir}. "
                         f"Required: {required_gb:.1f} GB, Available: {available_gb:.1f} GB")
            sys.exit(1)
        logger.info(f"PRE-FLIGHT: Disk space OK — {available_gb:.1f} GB available, "
                     f"~{required_gb:.1f} GB required (1.1x safety)")
    except Exception as e:
        logger.warning(f"PRE-FLIGHT: Could not check disk space: {e}")


def preflight_stale_output_check(output_dir: str, logger):
    """[V10] Warn if output directory already has files from a previous run."""
    if not os.path.isdir(output_dir):
        return
    existing = [f for f in os.listdir(output_dir)
                if f.endswith((".jsonl", ".csv", ".json", ".log"))]
    if existing:
        logger.warning(f"PRE-FLIGHT WARNING: Output directory {output_dir} already contains "
                       f"{len(existing)} file(s) from a previous run. "
                       f"Files: {', '.join(existing[:10])}{'...' if len(existing) > 10 else ''}")
        logger.warning("Previous output files will be OVERWRITTEN.")


def preflight_csv_duplicate_check(file_entries: list, logger):
    """[V12] Check for duplicate file paths in the input CSV."""
    seen = {}
    duplicates = []
    for filepath, key in file_entries:
        if filepath in seen:
            duplicates.append(filepath)
        else:
            seen[filepath] = key

    if duplicates:
        logger.error(f"PRE-FLIGHT FAILED: {len(duplicates)} duplicate path(s) in input CSV:")
        for d in duplicates[:20]:
            logger.error(f"  DUPLICATE: {d}")
        logger.error("Remove duplicates from CSV and re-run.")
        sys.exit(1)
    logger.info(f"PRE-FLIGHT: No duplicate paths in CSV ({len(file_entries)} unique files)")


# ═══════════════════════════════════════════════════════════════════════════
# INLINE VALIDATION (within process_part)
# ═══════════════════════════════════════════════════════════════════════════
# [V1] Row accounting: kept + rejected == total_rows (checked in process_part)
# [V5] Empty line counting: tracked in stats


# ═══════════════════════════════════════════════════════════════════════════
# POST-MERGE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def count_file_lines(filepath: str) -> int:
    """Count lines in a file efficiently.
    Handles files that don't end with a trailing newline."""
    count = 0
    last_byte = b""
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
            last_byte = chunk[-1:] if chunk else b""
    # If file is non-empty and doesn't end with \n, there's one more line
    if last_byte and last_byte != b"\n":
        count += 1
    return count


def verify_row_accounting(all_stats: list, logger) -> list:
    """[V1] Verify kept + rejected + unknown == total_rows for each part."""
    errors = []
    for s in all_stats:
        expected = s["total_rows"]
        actual = s["kept_rows"] + s["rejected_rows"] + s["unknown_rows"]
        if expected != actual:
            msg = (f"ROW ACCOUNTING MISMATCH: {s['filename']} part {s['part_idx']} — "
                   f"total={expected}, kept+rejected+unknown={actual} "
                   f"(diff={expected - actual})")
            logger.error(msg)
            errors.append(msg)
    if not errors:
        logger.info("VERIFY: Row accounting OK (kept + rejected + unknown == total_rows for all parts)")
    return errors


def verify_output_line_counts(by_file: dict, output_dir: str, logger,
                              merged_basenames: set = None) -> list:
    """[V2] Verify merged output file line counts match stats."""
    errors = []
    for filepath in sorted(by_file.keys()):
        parts = by_file[filepath]
        basename = parts[0]["basename"]

        # Skip files that weren't merged
        if merged_basenames is not None and basename not in merged_basenames:
            continue

        expected_en = sum(p["kept_rows"] for p in parts)
        expected_non = sum(p["rejected_rows"] for p in parts)
        expected_unk = sum(p["unknown_rows"] for p in parts)

        en_path = os.path.join(output_dir, f"{basename}_en.jsonl")
        non_path = os.path.join(output_dir, f"{basename}_non_selected.jsonl")
        unk_path = os.path.join(output_dir, f"{basename}_unknown.jsonl")

        if os.path.exists(en_path):
            actual_en = count_file_lines(en_path)
            if actual_en != expected_en:
                msg = (f"LINE COUNT MISMATCH: {basename}_en.jsonl — "
                       f"expected {expected_en}, got {actual_en}")
                logger.error(msg)
                errors.append(msg)

        if os.path.exists(non_path):
            actual_non = count_file_lines(non_path)
            if actual_non != expected_non:
                msg = (f"LINE COUNT MISMATCH: {basename}_non_selected.jsonl — "
                       f"expected {expected_non}, got {actual_non}")
                logger.error(msg)
                errors.append(msg)

        if os.path.exists(unk_path):
            actual_unk = count_file_lines(unk_path)
            if actual_unk != expected_unk:
                msg = (f"LINE COUNT MISMATCH: {basename}_unknown.jsonl — "
                       f"expected {expected_unk}, got {actual_unk}")
                logger.error(msg)
                errors.append(msg)

    if not errors:
        logger.info("VERIFY: Output line counts OK (all merged files match stats)")
    return errors


def verify_byte_range_coverage(all_stats: list, file_entries: list, logger) -> list:
    """[V6] Verify sum of byte ranges equals file size for each file."""
    errors = []
    by_file = defaultdict(list)
    for s in all_stats:
        by_file[s["filepath"]].append(s)

    file_sizes = {fp: os.path.getsize(fp) for fp, _ in file_entries}

    for filepath, parts in by_file.items():
        parts_sorted = sorted(parts, key=lambda s: s["part_idx"])
        total_bytes = sum(p["end_byte"] - p["start_byte"] for p in parts_sorted)
        actual_size = file_sizes.get(filepath, 0)

        if total_bytes != actual_size:
            msg = (f"BYTE RANGE GAP/OVERLAP: {os.path.basename(filepath)} — "
                   f"file_size={actual_size}, sum(byte_ranges)={total_bytes}, "
                   f"diff={actual_size - total_bytes}")
            logger.error(msg)
            errors.append(msg)

        # Check for gaps between parts
        for i in range(1, len(parts_sorted)):
            prev_end = parts_sorted[i - 1]["end_byte"]
            curr_start = parts_sorted[i]["start_byte"]
            if prev_end != curr_start:
                msg = (f"BYTE RANGE DISCONTINUITY: {os.path.basename(filepath)} — "
                       f"part {i-1} ends at {prev_end}, part {i} starts at {curr_start}")
                logger.error(msg)
                errors.append(msg)

    if not errors:
        logger.info("VERIFY: Byte range coverage OK (no gaps/overlaps)")
    return errors


def verify_output_size_sanity(by_file: dict, output_dir: str, file_entries: list,
                              logger, merged_basenames: set = None) -> list:
    """[V7] Verify output file sizes are roughly consistent with input."""
    errors = []
    file_sizes = {fp: os.path.getsize(fp) for fp, _ in file_entries}

    for filepath in sorted(by_file.keys()):
        parts = by_file[filepath]
        basename = parts[0]["basename"]

        # Skip files that weren't merged
        if merged_basenames is not None and basename not in merged_basenames:
            continue

        input_size = file_sizes.get(filepath, 0)

        en_path = os.path.join(output_dir, f"{basename}_en.jsonl")
        non_path = os.path.join(output_dir, f"{basename}_non_selected.jsonl")
        unk_path = os.path.join(output_dir, f"{basename}_unknown.jsonl")

        output_size = 0
        if os.path.exists(en_path):
            output_size += os.path.getsize(en_path)
        if os.path.exists(non_path):
            output_size += os.path.getsize(non_path)
        if os.path.exists(unk_path):
            output_size += os.path.getsize(unk_path)

        if input_size > 0:
            ratio = output_size / input_size
            # Allow 5% tolerance (some lines may be empty/skipped)
            if ratio < 0.90:
                msg = (f"OUTPUT SIZE WARNING: {basename} — "
                       f"input={input_size:,} bytes, output(en+non_sel)={output_size:,} bytes, "
                       f"ratio={ratio:.3f} (<0.90 — possible data loss)")
                logger.warning(msg)
                errors.append(msg)
            elif ratio > 1.05:
                msg = (f"OUTPUT SIZE WARNING: {basename} — "
                       f"input={input_size:,} bytes, output(en+non_sel)={output_size:,} bytes, "
                       f"ratio={ratio:.3f} (>1.05 — possible duplication)")
                logger.warning(msg)
                errors.append(msg)

    if not errors:
        logger.info("VERIFY: Output size sanity OK (en + non_selected ≈ input for all files)")
    return errors


def verify_boundary_no_duplicates(by_file_stats: dict, output_dir: str, logger) -> list:
    """[V3] Verify no row duplication at part boundaries.
    For each multi-part file, read the actual input file at boundary byte offsets
    and verify the line-skip logic didn't cause the same line to be read by two parts.
    
    Method: For each boundary between part N and part N+1, read the input file at
    part N's end_byte. If end_byte falls mid-line, both parts might process it.
    We verify by checking that end_byte of part N == start_byte of part N+1
    (already covered by V6) AND that the byte just before start_byte of part N+1
    is a newline (meaning the skip logic worked correctly).
    """
    errors = []

    for filepath, parts in by_file_stats.items():
        if len(parts) <= 1:
            continue

        parts_sorted = sorted(parts, key=lambda s: s["part_idx"])

        try:
            with open(filepath, "rb") as f:
                for i in range(1, len(parts_sorted)):
                    boundary = parts_sorted[i]["start_byte"]
                    if boundary == 0:
                        continue

                    # The byte just before start_byte should be \n
                    # (because find_line_boundary aligned to newline + 1)
                    f.seek(boundary - 1)
                    byte_before = f.read(1)
                    if byte_before != b"\n":
                        msg = (f"BOUNDARY ALIGNMENT ERROR: {os.path.basename(filepath)} — "
                               f"part {i} start_byte={boundary}, but byte at {boundary-1} "
                               f"is {byte_before!r}, not newline. "
                               f"Line-skip logic may have caused duplication or loss.")
                        logger.error(msg)
                        errors.append(msg)
        except Exception as e:
            logger.warning(f"Could not verify boundaries for {os.path.basename(filepath)}: {e}")

    if not errors:
        logger.info("VERIFY: Boundary alignment OK (all part boundaries fall on newlines)")
    return errors


def verify_json_sample(output_dir: str, by_file_stats: dict, sample_rate: float,
                       logger) -> list:
    """[V4] Re-parse a random sample of output rows to verify valid JSON."""
    errors = []
    total_sampled = 0
    total_invalid = 0

    for filepath, parts in by_file_stats.items():
        basename = parts[0]["basename"]
        en_path = os.path.join(output_dir, f"{basename}_en.jsonl")

        if not os.path.exists(en_path) or os.path.getsize(en_path) == 0:
            continue

        with open(en_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if random.random() > sample_rate:
                    continue
                total_sampled += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    json_loads(line)
                except (ValueError, json.JSONDecodeError):
                    total_invalid += 1
                    if total_invalid <= 5:
                        logger.error(f"INVALID JSON in {basename}_en.jsonl: "
                                     f"{line[:100]}...")

    if total_invalid > 0:
        msg = f"JSON VALIDATION: {total_invalid}/{total_sampled} sampled rows are invalid JSON"
        logger.error(msg)
        errors.append(msg)
    elif total_sampled > 0:
        logger.info(f"VERIFY: JSON sample validation OK ({total_sampled} rows sampled, all valid)")
    else:
        logger.info("VERIFY: JSON sample validation skipped (no rows sampled)")

    return errors


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Language filter for JSONL files — keep English only only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python language_filter.py --file-list inputs.csv --output /data/filtered --model-path lid.176.ftz
  python language_filter.py --file-list inputs.csv --output /out --workers 16 --split-threshold 512
  python language_filter.py --file-list inputs.csv --output /out --confidence 0.9 --default-key text
        """,
    )
    parser.add_argument("--file-list", required=True,
                        help="CSV file with (path, key) columns")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=8,
                        help="Number of worker processes (default: 8)")
    parser.add_argument("--split-threshold", type=int, default=512,
                        help="File split threshold in MB (default: 512)")
    parser.add_argument("--confidence", type=float, default=0.7,
                        help="Fasttext confidence threshold (default: 0.7)")
    parser.add_argument("--model-path", default="lid.176.ftz",
                        help="Path to fasttext language ID model (default: lid.176.ftz)")
    parser.add_argument("--chunk-size", type=int, default=100000,
                        help="Chars per chunk for long-text detection (default: 100000)")
    parser.add_argument("--max-chunks", type=int, default=20,
                        help="Max chunks for long-text detection (default: 20)")
    parser.add_argument("--min-en-ratio", type=float, default=0.9,
                        help="Min fraction of chunks that must be English to keep (default: 0.9)")
    parser.add_argument("--default-key", default=None,
                        help="Default JSON key when CSV row has no key specified")
    parser.add_argument("--enable-code-detection", action="store_true",
                        help="Enable code detection heuristic (default: off)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from previous run — skip files already in manifest.json (default: on)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Force reprocess all files, ignore previous manifest")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without processing — file list, work units, estimates")
    args = parser.parse_args()

    output_base = os.path.abspath(args.output)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, f"run_{run_timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger, log_dir = setup_logging(output_dir, run_timestamp)

    split_threshold_bytes = args.split_threshold * 1024 * 1024

    # ── Parse input CSV ──
    logger.info(f"Parsing file list: {args.file_list}")
    file_entries = parse_file_list(args.file_list, args.default_key)
    if not file_entries:
        logger.error("No .jsonl files found from input CSV. Exiting.")
        sys.exit(1)

    logger.info(f"Discovered {len(file_entries)} .jsonl file(s)")

    # ── Pre-flight checks ──
    preflight_fasttext_check(args.model_path, logger)                # [V11]
    preflight_csv_duplicate_check(file_entries, logger)         # [V12]

    # Check for output basename collisions
    basename_map = defaultdict(list)
    for fp, _ in file_entries:
        bn = make_output_basename(fp)
        basename_map[bn].append(fp)
    collisions = {bn: fps for bn, fps in basename_map.items() if len(fps) > 1}
    if collisions:
        logger.error("PRE-FLIGHT FAILED: Output basename collisions detected:")
        for bn, fps in collisions.items():
            logger.error(f"  '{bn}' ← {fps}")
        logger.error("Rename or reorganize input files to avoid collisions.")
        sys.exit(1)
    logger.info(f"PRE-FLIGHT: No output basename collisions")

    preflight_stale_output_check(output_dir, logger)            # [V10]

    # ── Resume check ──
    skip_files = set()
    if args.resume and not args.no_resume:
        skip_files = check_resume(output_dir, output_base, file_entries, args, logger)
        if skip_files:
            file_entries = [(fp, key) for fp, key in file_entries if fp not in skip_files]
            if not file_entries:
                logger.info("All files already processed. Nothing to do.")
                sys.exit(0)

    # ── Per-file key validation (head + tail check) ──
    validated_entries = []
    for filepath, key in file_entries:
        if not _validate_file_key(filepath, key, logger):
            logger.warning(f"SKIP: {os.path.basename(filepath)} — key '{key}' not found "
                           f"in first/last 10 rows, skipping this file")
            continue
        validated_entries.append((filepath, key))

    if not validated_entries:
        logger.error("No files passed key validation. Exiting.")
        sys.exit(1)

    if len(validated_entries) < len(file_entries):
        logger.info(f"Key validation: {len(validated_entries)}/{len(file_entries)} files passed")

    file_entries = validated_entries

    total_input_bytes = sum(os.path.getsize(fp) for fp, _ in file_entries)
    preflight_disk_space_check(output_dir, total_input_bytes, logger)  # [V9]

    # ── Select parts directory (RAM vs disk) ──
    parts_dir, parts_in_ram = select_parts_dir(
        output_dir, total_input_bytes, run_timestamp, logger=logger)

    # ── Build work queue ──
    tasks = []
    for filepath, key in file_entries:
        file_size = os.path.getsize(filepath)
        parts = split_file_into_parts(filepath, file_size, split_threshold_bytes)
        for (fp, sb, eb, pidx, tparts) in parts:
            tasks.append({
                "filepath": fp,
                "start_byte": sb,
                "end_byte": eb,
                "part_idx": pidx,
                "total_parts": tparts,
                "key": key,
                "output_dir": output_dir,
                "parts_dir": parts_dir,
                "confidence": args.confidence,
                "chunk_size": args.chunk_size,
                "max_chunks": args.max_chunks,
                "min_en_ratio": args.min_en_ratio,
                "enable_code_detection": args.enable_code_detection,
            })

    # ── Print banner ──
    total_size_gb = sum(os.path.getsize(fp) for fp, _ in file_entries) / (1024**3)
    logger.info("=" * 65)
    logger.info("  Language Filter — Keep English only")
    logger.info("=" * 65)

    # Node info
    try:
        hostname = socket.gethostname()
        node_ip = socket.gethostbyname(hostname)
        logger.info(f"  Node           : {hostname} ({node_ip})")
    except Exception:
        logger.info(f"  Node           : {os.uname().nodename}")

    logger.info(f"  Files          : {len(file_entries)}")
    logger.info(f"  Total size     : {total_size_gb:.2f} GB")
    logger.info(f"  Work units     : {len(tasks)}")
    logger.info(f"  Workers        : {args.workers}")
    logger.info(f"  Split threshold: {args.split_threshold} MB")
    logger.info(f"  Confidence     : {args.confidence}")
    logger.info(f"  Model          : {args.model_path}")
    logger.info(f"  Chunk size     : {args.chunk_size}")
    logger.info(f"  Max chunks     : {args.max_chunks}")
    logger.info(f"  Min EN ratio   : {args.min_en_ratio}")
    logger.info(f"  Code detection : {'ON' if args.enable_code_detection else 'OFF'}")
    logger.info(f"  JSON library   : {_JSON_LIB}")
    logger.info(f"  Default key    : {args.default_key or '(none)'}")
    logger.info(f"  Output         : {output_dir}")
    logger.info(f"  Parts dir      : {parts_dir} ({'RAM' if parts_in_ram else 'disk'})")
    resume_active = args.resume and not args.no_resume
    logger.info(f"  Resume         : {'ON' if resume_active else 'OFF'}"
                f"{f' (skipped {len(skip_files)} file(s))' if skip_files else ''}")
    logger.info("=" * 65)

    # ── Dry run ──
    if args.dry_run:
        # Estimate processing time based on ~8000 rows/sec with 160+ workers
        total_rows_est = total_input_bytes / 73000  # avg ~73KB per row from slimpajama
        est_time_sec = total_rows_est / max(args.workers * 45, 1)  # ~45 rows/sec/worker
        logger.info("")
        logger.info("DRY RUN — no processing will be done")
        logger.info(f"  Estimated rows : ~{int(total_rows_est):,}")
        logger.info(f"  Estimated time : ~{est_time_sec:.0f}s ({est_time_sec/60:.1f} min)")
        logger.info(f"  Disk needed    : ~{total_input_bytes * 1.1 / (1024**3):.1f} GB")
        logger.info(f"  RAM for parts  : ~{total_input_bytes * 1.1 / (1024**3):.1f} GB")
        logger.info("")
        logger.info("  Files to process:")
        for fp, key in file_entries:
            sz = os.path.getsize(fp)
            nparts = max(1, (sz + split_threshold_bytes - 1) // split_threshold_bytes)
            logger.info(f"    {os.path.basename(fp)} — {sz / (1024**3):.2f} GB, "
                        f"key='{key}', ~{nparts} parts")
        logger.info("")
        logger.info("Remove --dry-run to start processing.")
        # Clean up parts dir created during setup
        if os.path.isdir(parts_dir):
            shutil.rmtree(parts_dir)
        sys.exit(0)

    # ── Process with worker pool ──
    t0 = time.time()
    all_stats = []

    # Start RAM monitor if using RAM-backed parts
    ram_monitor = None
    if parts_in_ram:
        ram_monitor = RAMMonitor(parts_dir, threshold=0.90, interval=30, logger=logger)
        ram_monitor.start()

    # Progress logger thread — logs to file every 30s for remote monitoring
    progress_state = {"completed": 0, "kept": 0, "rejected": 0, "unknown": 0,
                      "total": len(tasks), "start_time": time.time()}
    progress_stop = threading.Event()

    def _progress_logger():
        while not progress_stop.is_set():
            progress_stop.wait(30)
            if progress_stop.is_set():
                break
            s = progress_state
            elapsed = time.time() - s["start_time"]
            pct = s["completed"] / s["total"] * 100 if s["total"] > 0 else 0
            rate = s["completed"] / elapsed if elapsed > 0 else 0
            eta = (s["total"] - s["completed"]) / rate if rate > 0 else 0
            logger.info(
                f"PROGRESS: {s['completed']}/{s['total']} parts ({pct:.0f}%) — "
                f"kept={s['kept']:,}, rejected={s['rejected']:,}, unknown={s['unknown']:,} — "
                f"{rate:.1f} parts/sec, ETA {eta:.0f}s")

    progress_thread = threading.Thread(target=_progress_logger, daemon=True)
    progress_thread.start()

    with mp.Pool(processes=args.workers, initializer=worker_init,
                 initargs=(args.model_path,)) as pool:
        failed_tasks = []
        results_iter = pool.imap_unordered(process_part_safe, tasks)

        # Wrap with tqdm if available
        if HAS_TQDM:
            pbar = tqdm(total=len(tasks), desc="Processing", unit="part",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                   "[{elapsed}<{remaining}, {rate_fmt}]")
            # Suppress per-part INFO on console to avoid garbling tqdm
            # (per-part details still go to log files)
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                    handler.setLevel(logging.WARNING)
                    break
        else:
            pbar = None

        total_kept = 0
        total_rejected = 0
        total_unknown = 0

        for i, result in enumerate(results_iter, 1):
            if result.get("_error"):
                failed_tasks.append(result)
                logger.error(
                    f"WORKER FAILED: "
                    f"{result['filename']} part {result['part_idx']}/{result['total_parts']} — "
                    f"{result['_error_msg']}"
                )
            else:
                all_stats.append(result)
                total_kept += result["kept_rows"]
                total_rejected += result["rejected_rows"]
                total_unknown += result["unknown_rows"]
                part_info = ""
                if result["total_parts"] > 1:
                    part_info = f" part {result['part_idx']}/{result['total_parts']}"
                logger.info(
                    f"[{i}/{len(tasks)}] {result['filename']}{part_info} — "
                    f"rows={result['total_rows']}, kept={result['kept_rows']}, "
                    f"rejected={result['rejected_rows']}"
                )

            if pbar:
                pbar.set_postfix(kept=f"{total_kept:,}", rejected=f"{total_rejected:,}",
                                 refresh=False)
                pbar.update(1)

            # Update progress for log thread
            progress_state["completed"] = len(all_stats) + len(failed_tasks)
            progress_state["kept"] = total_kept
            progress_state["rejected"] = total_rejected
            progress_state["unknown"] = total_unknown

        if pbar:
            pbar.close()
            # Restore console handler level
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                    handler.setLevel(logging.INFO)
                    break

    # Stop progress logger
    progress_stop.set()
    progress_thread.join(timeout=5)

    if failed_tasks:
        logger.error(f"CRITICAL: {len(failed_tasks)} work unit(s) failed. "
                     f"Output will be INCOMPLETE.")

    if not all_stats:
        logger.error("All workers failed. No output produced.")
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info(f"Processing complete in {elapsed:.1f}s")

    # ── Merge part files per source file ──
    logger.info("Merging part files...")
    by_file = defaultdict(list)
    for s in all_stats:
        by_file[s["filepath"]].append(s)

    merge_errors = []
    merged_basenames = set()

    for filepath in sorted(by_file.keys()):
        parts = sorted(by_file[filepath], key=lambda s: s["part_idx"])
        basename = parts[0]["basename"]

        # [FIX] Pre-merge verification: check each part file has expected row count
        parts_ok = True
        for p in parts:
            for label, path_key, expected_key in [
                ("en", "en_path", "kept_rows"),
                ("non_selected", "non_sel_path", "rejected_rows"),
                ("unknown", "unknown_path", "unknown_rows"),
            ]:
                part_path = p[path_key]
                expected_lines = p[expected_key]
                if not os.path.exists(part_path):
                    logger.error(f"MERGE ABORT: Missing part file {part_path}")
                    merge_errors.append(f"Missing: {part_path}")
                    parts_ok = False
                    continue
                actual_lines = count_file_lines(part_path)
                if actual_lines != expected_lines:
                    logger.error(
                        f"MERGE INTEGRITY: {os.path.basename(part_path)} — "
                        f"expected {expected_lines} lines, got {actual_lines} "
                        f"(possible truncation/disk full)")
                    merge_errors.append(
                        f"Truncated: {part_path} expected={expected_lines} actual={actual_lines}")
                    parts_ok = False

        if not parts_ok:
            logger.error(f"SKIPPING merge for {basename} due to integrity errors")
            continue

        # Merge en parts
        en_hi_parts = [p["en_path"] for p in parts]
        merged_en = os.path.join(output_dir, f"{basename}_en.jsonl")
        merge_part_files(en_hi_parts, merged_en)

        # Merge non_selected parts
        non_sel_parts = [p["non_sel_path"] for p in parts]
        merged_non_sel = os.path.join(output_dir, f"{basename}_non_selected.jsonl")
        merge_part_files(non_sel_parts, merged_non_sel)

        # Merge unknown parts
        unknown_parts = [p["unknown_path"] for p in parts]
        merged_unknown = os.path.join(output_dir, f"{basename}_unknown.jsonl")
        merge_part_files(unknown_parts, merged_unknown)

        logger.info(f"  Merged {len(parts)} part(s) → {basename}_en.jsonl, "
                     f"{basename}_non_selected.jsonl, {basename}_unknown.jsonl")
        merged_basenames.add(basename)

    if merge_errors:
        logger.error(f"MERGE: {len(merge_errors)} integrity error(s) — "
                     f"some files were NOT merged")

    # ── Stats ──
    per_file, global_totals = aggregate_stats(all_stats, file_entries)

    # [14] Run fingerprint
    run_fingerprint = generate_run_fingerprint(args, file_entries)
    run_info = {
        "fingerprint": run_fingerprint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_clock_sec": round(elapsed, 2),
        "config": {
            "file_list": os.path.abspath(args.file_list),
            "output": output_dir,
            "workers": args.workers,
            "split_threshold_mb": args.split_threshold,
            "confidence": args.confidence,
            "model_path": os.path.abspath(args.model_path),
            "chunk_size": args.chunk_size,
            "max_chunks": args.max_chunks,
            "min_en_ratio": args.min_en_ratio,
            "default_key": args.default_key,
            "code_detection": args.enable_code_detection,
        },
    }

    csv_path = os.path.join(output_dir, "manifest.csv")
    json_path = os.path.join(output_dir, "manifest.json")
    write_stats_csv(per_file, global_totals, csv_path)
    write_manifest_json(per_file, global_totals, run_info, json_path)

    # Copy manifests to log dir for archival
    shutil.copy2(csv_path, os.path.join(log_dir, "manifest.csv"))
    shutil.copy2(json_path, os.path.join(log_dir, "manifest.json"))

    # ── Post-merge verification ──
    logger.info("-" * 65)
    logger.info("  VERIFICATION")
    logger.info("-" * 65)
    all_verify_errors = []

    # [V1] Row accounting
    all_verify_errors.extend(verify_row_accounting(all_stats, logger))

    # [V2] Output line counts vs stats (only for merged files)
    all_verify_errors.extend(verify_output_line_counts(by_file, output_dir, logger,
                                                        merged_basenames))

    # [V6] Byte range coverage
    all_verify_errors.extend(verify_byte_range_coverage(all_stats, file_entries, logger))

    # [V7] Output size sanity (only for merged files)
    all_verify_errors.extend(verify_output_size_sanity(by_file, output_dir, file_entries,
                                                        logger, merged_basenames))

    # [V3] Boundary duplicate spot-check (needs parts dir — before cleanup)
    # Build dict with part paths for boundary check
    by_file_parts = defaultdict(list)
    for s in all_stats:
        by_file_parts[s["filepath"]].append(s)
    all_verify_errors.extend(verify_boundary_no_duplicates(by_file_parts, output_dir, logger))

    # [V4] JSON re-parse sample (0.1% of output rows)
    all_verify_errors.extend(verify_json_sample(output_dir, by_file, sample_rate=0.001, logger=logger))

    if all_verify_errors:
        logger.warning(f"VERIFICATION: {len(all_verify_errors)} issue(s) found")
    else:
        logger.info("VERIFICATION: All checks passed")
    logger.info("-" * 65)

    # ── Final summary ──
    logger.info("=" * 65)
    logger.info("  SUMMARY")
    logger.info("=" * 65)
    logger.info(f"  Run fingerprint : {run_fingerprint}")
    logger.info(f"  Total files     : {global_totals['total_files']}")
    logger.info(f"  Total input     : {global_totals['total_input_bytes'] / (1024**3):.2f} GB")
    logger.info(f"  Total rows      : {global_totals['total_rows']:,}")
    logger.info(f"  Kept rows       : {global_totals['kept_rows']:,} ({global_totals['kept_pct']})")
    logger.info(f"  Rejected rows   : {global_totals['rejected_rows']:,}")
    logger.info(f"  Unknown lang    : {global_totals['unknown_rows']:,}")
    logger.info(f"  Code detected   : {global_totals['code_rows']:,}")
    logger.info(f"  Mixed language  : {global_totals['mixed_lang_rows']:,}")
    logger.info(f"  Contaminated   : {global_totals['contaminated_rows']:,}")
    logger.info(f"  JSON errors     : {global_totals['json_errors']:,}")
    logger.info(f"  Key missing     : {global_totals['key_missing']:,}")
    logger.info(f"  Content empty   : {global_totals['content_empty']:,}")
    logger.info(f"  Empty lines     : {global_totals['empty_lines']:,}")
    logger.info(f"  ---")
    logger.info(f"  Kept words      : {global_totals['kept_word_count']:,}")
    logger.info(f"  Rejected words  : {global_totals['rejected_word_count']:,}")
    logger.info(f"  Kept est tokens : {global_totals['kept_est_tokens']:,}")
    logger.info(f"  Rejected est tk : {global_totals['rejected_est_tokens']:,}")
    logger.info(f"  ---")
    logger.info(f"  Retention (rows): {global_totals['retention_ratio_rows']}%")
    logger.info(f"  Retention (wrds): {global_totals['retention_ratio_words']}%")
    logger.info(f"  Retention (tok) : {global_totals['retention_ratio_est_tokens']}%")
    logger.info(f"  ---")
    if global_totals["top_rejected_langs"]:
        top3 = list(global_totals["top_rejected_langs"].items())[:5]
        logger.info(f"  Top rejected    : {', '.join(f'{l}={c:,}' for l, c in top3)}")
    if global_totals["schema_counts"]:
        logger.info(f"  Schema dist     : {global_totals['schema_counts']}")
    logger.info(f"  ---")
    logger.info(f"  Wall clock      : {elapsed:.1f}s")
    logger.info(f"  CPU time (sum)  : {global_totals['total_processing_time_sec']:.1f}s")
    logger.info(f"  Manifest CSV    : {csv_path}")
    logger.info(f"  Manifest JSON   : {json_path}")
    logger.info(f"  ---")
    logger.info(f"  INTEGRITY CHECK:")
    total_rows = global_totals['total_rows']
    sum_rows = global_totals['kept_rows'] + global_totals['rejected_rows'] + global_totals['unknown_rows']
    logger.info(f"  kept + rejected + unknown = {global_totals['kept_rows']:,} + "
                f"{global_totals['rejected_rows']:,} + {global_totals['unknown_rows']:,} = "
                f"{sum_rows:,} {'== ' if sum_rows == total_rows else '!= '}{total_rows:,} "
                f"{'✓' if sum_rows == total_rows else '✗ MISMATCH'}")
    logger.info(f"  Verify issues   : {len(all_verify_errors)} "
                f"{'✓' if not all_verify_errors else '✗ SEE ABOVE'}")
    if all_verify_errors:
        for err in all_verify_errors[:5]:
            logger.info(f"    → {err}")
    logger.info("=" * 65)

    # ── Cleanup part files ──
    if os.path.isdir(parts_dir):
        shutil.rmtree(parts_dir)
        logger.info(f"  Cleaned up parts directory: {parts_dir} ({'RAM' if parts_in_ram else 'disk'})")

    # Stop RAM monitor
    if ram_monitor:
        ram_monitor.stop()

    logger.info("Done.")


if __name__ == "__main__":
    main()
