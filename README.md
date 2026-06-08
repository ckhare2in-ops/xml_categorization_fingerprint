# 3-Stage XML Error Classification Engine

Normalizes raw error messages from multiple customers into a single, consistent
taxonomy of 16 canonical categories — so that two customers reporting the "same"
problem in completely different wording land on the **same** category.

## How it works

```
Customer Raw Error
       │
       ▼
Stage 1 — Normalize (LLM, Groq)
   Strip customer-specific values → extract a structural "intent fingerprint"
       │
       ▼
Stage 2 — Embed (sentence-transformers, all-MiniLM-L6-v2)
   Fingerprint → embedding vector (for similarity search)
       │
       ▼
Stage 3 — Classify (LLM + fixed 16-category taxonomy)
   Pick EXACTLY one canonical category
       │
       ▼
Canonical Category (consistent across all customers)
```

- **Stage 1** ([`normalize_error`](engine.py)) calls the Groq LLM to extract a
  fingerprint with keys `violation_type`, `constraint_kind`, `scope`, `actor`.
- **Stage 2** ([`embed_fingerprint`](engine.py)) serializes the fingerprint with
  sorted keys and embeds it with `all-MiniLM-L6-v2`.
- **Stage 3** ([`classify_error`](engine.py)) maps the fingerprint to one of the
  16 fixed categories in [`taxonomy.py`](taxonomy.py). New categories are
  **never** invented — anything that doesn't fit becomes `Unknown / Unclassified`.

## Setup

1. **Create a virtual environment and install dependencies** (Python 3.9+):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   > A `.venv/` in the project root is the supported way to work on this
   > project — it keeps `torch`, `sentence-transformers`, etc. isolated. It is
   > git-ignored. Re-activate it (`source .venv/bin/activate`) in each new shell.

2. **Configure your API key:**

   ```bash
   cp .env.example .env
   # then edit .env and set GROQ_API_KEY=...
   ```

   Get a key at <https://console.groq.com>.

## Usage

### CLI

```bash
python classify.py --error "your error here" --customer "customer_a"
```

Add `--similar` to also see the most similar previously-classified errors:

```bash
python classify.py --error "element <foo> not allowed here" --customer "customer_a" --similar
```

### CLI — batch mode (CSV in, CSV out)

Classify many errors at once. Each row is run through the **same**
`classify_pipeline` used by single mode:

```bash
python classify.py --input errors.csv --output classified_errors.csv
```

The input CSV must contain an `error_message` column and a `customer` column
(any other columns are preserved in the output). The output CSV is all original
columns plus `category` and `confidence`.

- Rows missing `error_message` or `customer`, or that fail to classify, are
  **skipped and logged to stderr** — the run continues.
- `confidence` is currently emitted blank: the pipeline returns an exact
  category name (temperature 0), not a probability, so there is no honest score
  to report yet. The column exists so downstream tooling has a stable schema.

**Sample input (`errors.csv`):**

```csv
error_message,customer,source_file
"cvc-complex-type.2.4.a: Invalid content was found starting with element 'para'.",customer_a,chapter1.xml
"document type does not allow element ""sect2"" here",customer_b,manual.sgml
"Element 'title' is missing required attribute 'id'",customer_a,front.xml
```

**Sample output (`classified_errors.csv`):**

```csv
error_message,customer,source_file,category,confidence
"cvc-complex-type.2.4.a: Invalid content was found starting with element 'para'.",customer_a,chapter1.xml,Structural Mismatch Error,
"document type does not allow element ""sect2"" here",customer_b,manual.sgml,Structural Mismatch Error,
"Element 'title' is missing required attribute 'id'",customer_a,front.xml,Attribute ID Is Missing,
```

### As a library

```python
from engine import classify_pipeline, find_similar_errors, add_few_shot_example

result = classify_pipeline("cvc-complex-type.2.4.a: Invalid content...", "customer_a")
print(result["category"])

# Strengthen the classifier at runtime with a confirmed example:
add_few_shot_example("Structural Mismatch Error", result["fingerprint"])

# Find similar past errors:
for hit in find_similar_errors("unexpected child element", top_k=3):
    print(hit["similarity"], hit["category"])
```

## Example: two customer formats → one category

Both of the following errors are phrased completely differently by two different
customers, yet both normalize to a structural content-model violation and
classify as **`Structural Mismatch Error`**.

### Customer A (Xerces / XSD validator wording)

```bash
python classify.py \
  --error "cvc-complex-type.2.4.a: Invalid content was found starting with element 'para'. One of '{title, abstract}' is expected at line 42." \
  --customer "customer_a"
```

```json
{
  "id": "…",
  "customer_id": "customer_a",
  "raw_error": "cvc-complex-type.2.4.a: Invalid content was found starting with element 'para'. One of '{title, abstract}' is expected at line 42.",
  "fingerprint": {
    "violation_type": "element_placement_violation",
    "constraint_kind": "content_model",
    "scope": "structural",
    "actor": "element"
  },
  "category": "Structural Mismatch Error",
  "embedding_dim": 384
}
```

### Customer B (DTD / nsgmls wording)

```bash
python classify.py \
  --error "document type does not allow element \"sect2\" here; missing one of \"para\", \"list\" start-tag" \
  --customer "customer_b"
```

```json
{
  "id": "…",
  "customer_id": "customer_b",
  "raw_error": "document type does not allow element \"sect2\" here; missing one of \"para\", \"list\" start-tag",
  "fingerprint": {
    "violation_type": "element_placement_violation",
    "constraint_kind": "content_model",
    "scope": "structural",
    "actor": "element"
  },
  "category": "Structural Mismatch Error",
  "embedding_dim": 384
}
```

> The exact fingerprint wording is produced by the LLM and may vary slightly run
> to run, but both customer formats converge on `Structural Mismatch Error`.

## Output files

| File | Purpose |
|------|---------|
| `errors.jsonl` | Every classified error, one JSON object per line. |
| `embeddings.npy` | Numpy matrix of all embedding vectors (row order matches the lines in `errors.jsonl` / `embeddings_meta.jsonl`). |
| `embeddings_meta.jsonl` | Per-embedding metadata index: `id`, `customer_id`, `category`. |
| `review_queue.jsonl` | Errors that returned `Unknown / Unclassified`, queued for human review. |

These are created automatically on first run and are git-ignored.

## Configuration

Defined at the top of [`engine.py`](engine.py):

| Setting | Value |
|---------|-------|
| LLM model | `llama-3.3-70b-versatile` |
| Embedding model | `all-MiniLM-L6-v2` |
| `max_tokens` | `200` |
| `temperature` | `0` |
| Retry policy | up to 3 attempts per LLM call, exponential backoff (1s, 2s, 4s) |

## Taxonomy governance

- The 16 categories live in [`taxonomy.py`](taxonomy.py) and are the single
  source of truth.
- The classifier always picks from this fixed list; it cannot create new
  categories.
- `add_few_shot_example(category, fingerprint)` lets you reinforce a category
  with new examples at runtime (rejects unknown category names).
