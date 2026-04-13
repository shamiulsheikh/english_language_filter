# Language Filter Pipeline

English-only language filter for JSONL pretraining data with mixed-language contamination detection.

## Run It

### 1. Install (one-time)

```bash
pip install fasttext-wheel orjson tqdm --break-system-packages
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

### 2. Create input CSV

```csv
path,key
/data/slimpajama.jsonl,text
/data/another_folder,messages
```

### 3. Run

```bash
python3 lang.py --file-list inputs.csv --output /output/dir/ --model-path lid.176.ftz --workers 16 --default-key text
```

### 4. Check what key your file uses

```bash
head -1 /path/to/your/file.jsonl | python3 -c "import sys,json; print(list(json.loads(sys.stdin.readline()).keys()))"
```

### 5. Full production run (180 workers, 84GB in ~3 min)

```bash
python3 lang.py --file-list inputs.csv --output /output/dir/ --model-path lid.176.ftz --workers 180 --split-threshold 128 --confidence 0.7 --min-en-ratio 0.9 --chunk-size 100000 --max-chunks 25 --default-key text
```

### 6. Dry run (preview without processing)

```bash
python3 lang.py --file-list inputs.csv --output /output/dir/ --model-path lid.176.ftz --workers 180 --split-threshold 128 --dry-run --default-key text
```

### 7. Force reprocess (ignore previous runs)

```bash
python3 lang.py --file-list inputs.csv --output /output/dir/ --model-path lid.176.ftz --workers 180 --no-resume --default-key text
```

---

## Output

```
output/
└── run_20260408_141307/
    ├── file_A_en.jsonl              # Confident English rows
    ├── file_A_non_selected.jsonl    # Rejected rows
    ├── file_A_unknown.jsonl         # Unclassifiable rows
    ├── manifest.csv
    ├── manifest.json
    └── logs/
        └── lang_filter_run_.../
            ├── language_filter.log
            ├── manifest.csv
            └── manifest.json
```

Each run gets its own timestamped directory. Simultaneous runs never collide.

---

## CLI Reference

| Flag | Default | What it does |
|------|---------|-------------|
| `--file-list` | required | CSV with (path, key) columns |
| `--output` | required | Output base directory |
| `--model-path` | `lid.176.ftz` | Path to fasttext model file |
| `--workers` | 8 | Number of parallel workers |
| `--split-threshold` | 512 | File split size in MB for parallelism |
| `--confidence` | 0.7 | Per-chunk fasttext confidence threshold |
| `--chunk-size` | 100000 | Characters per chunk for long texts |
| `--max-chunks` | 20 | Max chunks per row |
| `--min-en-ratio` | 0.9 | Min fraction of English chunks to keep a row |
| `--default-key` | none | Fallback key when CSV row has no key |
| `--enable-code-detection` | off | Turn on code detection heuristic |
| `--dry-run` | off | Show plan without processing |
| `--no-resume` | off | Force reprocess all files |

---

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `fasttext-wheel` | Language identification (C++ backend) | Yes |
| `lid.176.ftz` | Pre-trained model (176 languages, 917KB) | Yes |
| `orjson` | Fast JSON parsing (3-10x vs stdlib) | No (auto-fallback) |
| `tqdm` | Progress bar with ETA | No (auto-fallback) |

---

## How It Works

### What problem does this solve?

Pretraining datasets contain non-English rows and mixed-language documents where English content is contaminated with Chinese, Russian, French, etc. These degrade model quality. This pipeline filters them out.

### File splitting (`--split-threshold`)

Large JSONL files are split into byte-range chunks at newline boundaries so multiple workers can process them in parallel.

```
slimpajama.jsonl (84 GB) + --split-threshold 128
    → 672 work units of ~128MB each
    → 180 workers grab from queue
    → 2.5 minutes
```

Without splitting: 84GB = one worker = hours.

### Why chunking instead of passing whole text to fasttext?

fasttext on the full text returns **one answer** — the majority language. If a document is 70% English and 30% Chinese, fasttext says "en" with high confidence. The 30% Chinese contamination is completely invisible. You keep the row thinking it's clean English.

```
Full text (700K English + 300K Chinese) → fasttext → "en" 0.85 → KEPT ← wrong!
```

With chunking, each section is evaluated independently:

```
chunk 0: English → "en"
chunk 1: English → "en"  
chunk 2: Chinese → "zh"  ← caught!
chunk 3: English → "en"
chunk 4: English → "en"

en_ratio = 4/5 = 0.80 < 0.90 → REJECTED ← correct!
```

fasttext sums all character n-grams into one vector. The majority language's n-grams outvote the minority. There's no way to detect contamination from a single call — it's mathematically hidden in the averaged vector. Chunking is the only way to see what's actually in each region of the document.

### Chunked language detection (`--chunk-size`, `--max-chunks`)

For each row, the text is split into evenly-spaced chunks. fasttext runs on each chunk independently. This catches contamination that a single full-text call would miss.

```
Row with 600K chars:
    chunk 0: chars[0:100K]     → "en" (0.95)
    chunk 1: chars[115K:215K]  → "en" (0.92)
    chunk 2: chars[230K:330K]  → "zh" (0.88)  ← contamination found
    chunk 3: chars[345K:445K]  → "en" (0.91)
    chunk 4: chars[460K:560K]  → "en" (0.93)

    en_ratio = 4/5 = 0.80 < 0.90 → REJECTED
```

Without chunking, fasttext on the full 600K chars returns "en" with 0.85 confidence. The Chinese section is invisible. You silently keep contaminated data.

### Two thresholds

**`--confidence 0.7`** — per chunk: "is fasttext sure about THIS chunk?"

A chunk returns `en` with 0.5 confidence → below 0.7 → marked "uncertain", doesn't count as English.

**`--min-en-ratio 0.9`** — across all chunks: "is the WHOLE document English enough?"

20 chunks, 18 English, 2 Chinese → en_ratio = 0.90 → keep. 17 English, 3 Chinese → 0.85 → reject.

### Decision flow

```
Text < 10 chars       → _unknown.jsonl
Text < 10K chars      → single fasttext call
                         en + conf ≥ 0.7     → _en.jsonl
                         en + conf < 0.7     → _non_selected.jsonl
                         non-English         → _non_selected.jsonl
                         unknown             → _unknown.jsonl
Text ≥ 10K chars      → chunked detection
                         all chunks uncertain → _unknown.jsonl
                         en_ratio ≥ 0.9      → _en.jsonl
                         en_ratio < 0.9      → _non_selected.jsonl
```

### Input CSV format

```csv
path,key
/data/folder1,messages
/data/file.jsonl,text
/data/folder2,
/data/folder3
```

- `path` is folder → scans for .jsonl files (non-recursive)
- `path` is file → uses directly
- `key` empty → uses `--default-key`

**What is `key`?**

`key` tells the script which JSON field inside each JSONL row contains the text to run language detection on. Different datasets use different field names for the same thing:

```json
{"text": "The quick brown fox..."}                          → key = text
{"messages": [{"role": "user", "content": "Hello"}]}       → key = messages
{"conversations": [{"from": "human", "value": "Hello"}]}   → key = conversations
{"instruction": "Translate this", "output": "Bonjour"}      → key = instruction
```

The script doesn't know which field your file uses. You tell it via the CSV. To find out:

```bash
head -1 /your/file.jsonl | python3 -c "import sys,json; print(list(json.loads(sys.stdin.readline()).keys()))"
```

Output: `['text', 'meta', 'source']` → use `text`.

**How each key type is handled:**

- `key=text` (or any flat string field) → uses the string value directly
- `key=messages` → ChatML format, concatenates all `content` fields from the messages array
- `key=conversations` → ShareGPT format, concatenates all `value`/`content` fields
- Any other key → treats as flat string field

---

## Resume

On by default. No flag needed.

- Scans previous `run_*` directories for latest manifest
- If config (confidence, min_en_ratio, chunk_size, max_chunks) changed → reprocesses all
- If same config → checks each file: input size unchanged + all 3 output files exist + line counts match
- Copies verified outputs to new run directory, skips reprocessing
- Use `--no-resume` to force reprocess

---

## Pre-Flight Checks

Runs before processing. Fails fast with clear messages:

- fasttext model exists and works
- No duplicate paths in input CSV
- No output basename collisions
- Disk space sufficient (1.1x input)
- Key exists in first/last 10 rows of each file (shows available keys if wrong)

---

## Post-Processing Verification

Runs after processing. Flags any issues:

- Row accounting: kept + rejected + unknown == total (per part)
- Output line counts match stats
- Byte range coverage: no gaps or overlaps
- Boundary alignment: all splits fall on newlines
- Output size sanity: output ≈ input size
- JSON sample re-parse: random 0.1% of output rows are valid

---

## Manifest

Per-file stats in CSV + JSON:

- Rows: total, kept, rejected, unknown, json_errors, key_missing, content_empty
- Languages: per-language counts
- Contamination: mixed_lang_rows, contaminated_rows
- Volume: word counts, estimated tokens
- Row size: avg/min/max for kept/rejected
- Confidence: avg/min/max, bucket distribution
- EN ratio: histogram across chunked rows
- Schema: chatml/sharegpt/flat/other
- Performance: processing time, rows/sec, MB/sec
- Provenance: input path, size, mtime → output filenames
- Run fingerprint: SHA256 of config + file list

---

## RAM-Backed Parts

Worker output goes to `/dev/shm` (RAM) if available and usage < 90%. Falls back to disk. Background monitor checks RAM every 30 seconds. Stale dirs from crashed runs cleaned via PID-based detection.

---

## Performance Tuning

| Dataset Size | Workers | Split Threshold | Expected Time |
|-------------|---------|-----------------|---------------|
| < 10 GB | 8-16 | 512 MB | < 1 min |
| 10-100 GB | 32-64 | 256 MB | 1-5 min |
| 100-500 GB | 128-180 | 128 MB | 5-30 min |
| 500+ GB | 180 | 128 MB | 30-60 min |

---

## No Hardcoded Paths

All paths are CLI/CSV configurable. Only `/dev/shm` (standard Linux tmpfs) is used internally, with automatic disk fallback.

---

## Why Chunking? (Validated with Experiments)

### The problem with full-text detection

fasttext sums all character n-grams into one vector and classifies. When English and Chinese are mixed, the English n-grams dominate — **Chinese contamination is invisible**:

| Mix | fasttext full-text says | Chinese in top-5? |
|-----|------------------------|-------------------|
| 5% Chinese | en (0.96) | No |
| 10% Chinese | en (0.95) | No |
| 20% Chinese | en (0.92) | No |
| 30% Chinese | en (0.86) | No |
| 40% Chinese | en (0.76) | Yes (5th place) |
| 50% Chinese | en (0.61) | Yes (4th place) |
| 80% Chinese | **it** (0.24) | Yes — but fasttext says Italian, not English or Chinese |

At 30% Chinese contamination, fasttext returns English with 0.86 confidence. Chinese doesn't appear anywhere in the top-5 predictions. You'd keep a document that's 30% Chinese and never know.

At 80% Chinese, fasttext breaks entirely — it returns **Italian** as the top language. Neither English nor Chinese is #1.

### Other languages ARE visible (Chinese is uniquely invisible)

With 30% contamination from other languages, fasttext shows them in top-5:

| 30% contamination | Top-1 | Foreign in top-5? |
|-------------------|-------|-------------------|
| Russian | ru (0.55) | Yes — becomes #1 |
| French | en (0.55) | Yes — #2 |
| German | de (0.54) | Yes — becomes #1 |
| Japanese | ja (0.46) | Yes — becomes #1 |
| Arabic | ar (0.67) | Yes — becomes #1 |
| Korean | ko (0.78) | Yes — becomes #1 |
| **Chinese** | **en (0.87)** | **No** |

Chinese uses CJK characters whose n-gram hashes collide with Romance language features in the model. Other scripts (Cyrillic, Arabic, Hangul, Devanagari) produce distinctly different n-grams that disrupt the English signal clearly.

### Confidence alone can't distinguish contamination from noise

| 30% mixed with... | Confidence | Actually contaminated? |
|-------------------|-----------|----------------------|
| Chinese | en (0.86) | Yes |
| Code | en (0.95) | No — valid English with code |
| Math/equations | en (0.94) | No — valid English with math |
| Short sentences | en (0.97) | No — valid English |

A threshold that catches 30% Chinese (< 0.87) would also reject clean English documents containing code or math.

### Chunked detection solves this

| Mix | Full-text catches it? | Chunked catches it? |
|-----|----------------------|---------------------|
| 5% Chinese | No | No (acceptable) |
| 10% Chinese | No | No (borderline) |
| 15% Chinese | No | **Yes** |
| 20% Chinese | No | **Yes** |
| 30% Chinese | No | **Yes** |
| 40% Chinese | No | **Yes** |
| 50% Chinese | Yes (conf < 0.7) | **Yes** |

Chunked detection catches contamination at **15%**, full-text only at **50%**. A 3x improvement in sensitivity.

### Position doesn't matter for chunking

| 30% Chinese position | Full-text | Chunked en_ratio |
|----------------------|-----------|-----------------|
| At start of document | en (0.86) — misses it | 0.7 — catches it |
| At end of document | en (0.86) — misses it | 0.7 — catches it |
| In the middle | en (0.86) — misses it | 0.6 — catches it |

### Known limitation: scattered contamination

If non-English text is scattered throughout the document in small fragments (e.g., alternating 8K English / 3K Chinese paragraphs), each chunk contains enough English to classify as English. Both full-text and chunked detection miss it:

```
ZH scattered (30%): full=en(0.87)  chunked en_ratio=1.0  ← BOTH MISS IT
```

This is uncommon in real pretraining data — contamination tends to be contiguous sections, not interleaved paragraphs. But if your data has this pattern, neither approach catches it.

---

## Pitfalls and Warnings

### Confidence threshold — what you lose

| Text type | fasttext confidence | 0.7 | 0.8 | 0.9 |
|-----------|-------------------|-----|-----|-----|
| Clean English prose | 0.88 | KEEP | KEEP | **REJECT** |
| English with typos | 0.81 | KEEP | KEEP | **REJECT** |
| English slang/informal | 0.92 | KEEP | KEEP | KEEP |
| English + numbers/stats | 0.87 | KEEP | KEEP | **REJECT** |
| English technical (TCP/IP, TLS) | 0.75 | KEEP | **REJECT** | **REJECT** |
| List/bullet text | 0.35 | **REJECT** | **REJECT** | **REJECT** |
| English about code | 0.74 | KEEP | **REJECT** | **REJECT** |

**Recommendation: `--confidence 0.7`** is the sweet spot. At 0.8, you lose technical writing. At 0.9, you lose clean prose, text with numbers, and almost everything except casual English.

**Warning:** List-style text ("Requirements: Python 3.8+ NumPy 1.21+ CUDA 11.7") gets very low confidence at ANY threshold. These rows will always be rejected or go to unknown. This is a fasttext limitation — short fragmented text doesn't produce enough n-gram signal.

### False positives — what slips through as "English"

| Text | fasttext says | Actually |
|------|-------------|---------|
| Romanized Hindi ("Kaise ho bhai") | en (0.33) | Hindi in Latin script — rejected at 0.7 ✓ |
| Pinyin Chinese ("Ni hao shijie") | en (0.19) | Chinese in Latin script — rejected at 0.7 ✓ |
| Indonesian | id (0.92) | Correctly identified — rejected ✓ |
| Malay | id (0.91) | Classified as Indonesian — still rejected ✓ |

Romanized foreign languages get low confidence and are correctly rejected at 0.7. Not a concern.

**Real false positive risk:** documents that are technically English but shouldn't be in pretraining data (e.g., auto-generated text, SEO spam, machine-translated English). fasttext cannot detect quality — only language. Quality filtering requires a separate pipeline.

### Code detection (`--enable-code-detection`) — USE WITH CAUTION

**Off by default for a reason.** The heuristic has known issues:

**False positives (flags English text as code):**
```
"In this tutorial we will learn how to write Python functions. 
You define a function using the def keyword..."
→ Heuristic: CODE (because "def " and "function " appear in text)
→ Reality: English tutorial text, should go through normal language detection
```

**False negatives (misses actual code):**
```
SELECT u.name, COUNT(o.id) FROM users u JOIN orders o...  → NOT detected
{"model": "gpt-4", "temperature": 0.7, ...}               → NOT detected  
\begin{equation} \int_{0}^{\infty} e^{-x^2} dx            → NOT detected
```

**When enabled, code detection bypasses language detection entirely.** If a row triggers the code heuristic, it goes straight to `_en.jsonl` without checking what language the code comments are in. A Chinese programming tutorial with Python keywords would be kept as "English."

**What happens to code WITHOUT code detection enabled:**
- Pure code (Python, JS, HTML) gets confidence 0.17-0.41 → rejected as "uncertain"
- These go to `_unknown.jsonl`, not `_non_selected.jsonl`
- English text mixed with code gets 0.66 → rejected at 0.7, kept at 0.6

**Recommendation:** Keep code detection OFF unless you specifically need to preserve code-heavy documents. If you do enable it, review the `code_rows` count in the manifest and spot-check them.

### min-en-ratio strictness

| Threshold | What it means | Trade-off |
|-----------|--------------|-----------|
| 1.0 | Every single chunk must be English | Rejects docs with ANY non-English (headers, quotes, names) |
| 0.9 (default) | 90% of chunks must be English | Catches 15%+ contamination, allows minor non-English |
| 0.8 | 80% of chunks must be English | More permissive, allows 20% contamination |
| 0.5 | Half the chunks can be non-English | Very permissive, only catches majority non-English |

**Warning at 1.0:** English academic papers quoting foreign-language sources, or articles about foreign politics containing names in Cyrillic/Arabic, will be rejected. Don't use 1.0 unless your data is expected to be purely English with zero foreign words.

### Unknown rows — what are they?

Rows go to `_unknown.jsonl` when:
- Text is shorter than 10 characters (too short to detect)
- All chunks scored below confidence threshold (all "uncertain")
- Key exists but extracted text is empty

**These are NOT necessarily bad data.** They could be:
- Math-heavy content (equations, formulas)
- Code-heavy content (low natural language signal)
- Very short metadata rows
- Tabular data or structured content

Check the `unknown_rows` count in the manifest. If it's a large fraction of your data, inspect a sample before discarding.

### What this pipeline does NOT detect

- **English quality** — grammatically broken, machine-translated, or low-quality English passes if fasttext says "en"
- **Duplicate content** — identical or near-duplicate rows are not detected
- **Toxic/harmful content** — no content filtering
- **Domain relevance** — off-topic English content passes
- **Encoding issues** — mojibake (garbled encoding) may or may not be detected depending on the byte patterns

---

*Built by Sheikh Shamiul Huda*
