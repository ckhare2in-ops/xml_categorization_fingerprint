# 3-Stage XML Error Classification Engine

Normalizes raw error messages from multiple customers into a single, consistent
taxonomy of 16 canonical categories — so that two customers reporting the "same"
problem in completely different wording land on the **same** category.

Built on **LangChain**: the two LLM stages are LCEL chains over `ChatOpenAI`
(`gpt-4o-mini`), the embedding stage uses `OpenAIEmbeddings`
(`text-embedding-3-small`), and similarity search is backed by a LangChain
**FAISS** vector store. LLM calls are traced with **Langfuse**.

## How it works

```
Customer Raw Error
       │
       ▼
Stage 1 — Normalize (LangChain LCEL: ChatOpenAI gpt-4o-mini + JsonOutputParser)
   Strip customer-specific values → extract a structural "intent fingerprint"
       │
       ▼
Stage 2 — Embed (LangChain OpenAIEmbeddings, text-embedding-3-small)
   Fingerprint → embedding vector → FAISS vector store (for similarity search)
       │
       ▼
Stage 3 — Classify (LangChain LCEL: ChatOpenAI gpt-4o-mini + fixed taxonomy)
   Pick EXACTLY one canonical category
       │
       ▼
Canonical Category (consistent across all customers)
```

- **Stage 1** ([`normalize_error`](engine.py)) is an LCEL chain
  `ChatPromptTemplate | ChatOpenAI | JsonOutputParser | RunnableLambda` that
  extracts a fingerprint with keys `violation_type`, `constraint_kind`, `scope`,
  `actor`.
- **Stage 2** ([`embed_fingerprint`](engine.py)) serializes the fingerprint with
  sorted keys and embeds it with `OpenAIEmbeddings` (`text-embedding-3-small`,
  1536-dim, unit-normalized).
- **Stage 3** ([`classify_error`](engine.py)) is an LCEL chain
  `ChatPromptTemplate | ChatOpenAI | StrOutputParser | RunnableLambda` that maps
  the fingerprint to one of the 16 fixed categories in
  [`taxonomy.py`](taxonomy.py). New categories are **never** invented — anything
  that doesn't fit becomes `Unknown / Unclassified`.
- Each LLM chain is wrapped with LangChain's native `.with_retry(...)` (3
  attempts, exponential backoff with jitter) and traced via Langfuse (see
  [Observability](#observability-langfuse)).

## Setup

1. **Create a virtual environment and install dependencies** (Python 3.9+):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   > A `.venv/` in the project root is the supported way to work on this
   > project — it keeps the LangChain / FAISS deps isolated. It is git-ignored.
   > Re-activate it (`source .venv/bin/activate`) in each new shell.

2. **Configure your keys:**

   ```bash
   cp .env.example .env
   # then edit .env:
   #   OPENAI_API_KEY=...        (required — chat + embeddings)
   #   LANGFUSE_PUBLIC_KEY=...   (optional — tracing)
   #   LANGFUSE_SECRET_KEY=...
   ```

   Get an OpenAI key at <https://platform.openai.com/api-keys> and (optionally)
   Langfuse keys at <https://cloud.langfuse.com>. Without the `LANGFUSE_*` keys
   the pipeline runs normally and tracing is simply skipped.

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
  "embedding_dim": 1536
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
  "embedding_dim": 1536
}
```

> The exact fingerprint wording is produced by the LLM and may vary slightly run
> to run, but both customer formats converge on `Structural Mismatch Error`.

## Output files

| Store | Purpose |
|-------|---------|
| `errors.jsonl` | Human-readable append-only log of every classified error, one JSON object per line. |
| `faiss_index/` | LangChain FAISS vector store (`index.faiss` + `index.pkl`). The searchable record of every error, with the full record in each document's metadata; backs `find_similar_errors`. |
| `review_queue.jsonl` | Errors that returned `Unknown / Unclassified`, queued for human review. |

These are created automatically on first run and are git-ignored. The FAISS
vectors are 1536-dim (`text-embedding-3-small`); if you switch embedding models,
delete `faiss_index/` first to avoid a dimension mismatch.

## Configuration

Defined at the top of [`engine.py`](engine.py):

| Setting | Value |
|---------|-------|
| LLM | `ChatOpenAI`, model `gpt-4o-mini` |
| Embeddings | `OpenAIEmbeddings`, `text-embedding-3-small` (1536-dim, unit-normalized) |
| Vector store | LangChain `FAISS` (cosine via squared-L2 conversion) |
| Tracing | Langfuse LangChain `CallbackHandler` (optional) |
| `max_tokens` | `200` |
| `temperature` | `0` |
| Retry policy | `.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)` per chain |

## Observability (Langfuse)

The two LLM stages are traced with [Langfuse](https://langfuse.com) via its
LangChain `CallbackHandler`, attached per-invoke. Each `classify_pipeline` (and
`find_similar_errors`) call is wrapped with `@observe()`, so one call shows up as
**a single trace** in the Langfuse dashboard, with the `normalize_error` and
`classify_error` LLM generations nested under it.

- Tracing is **opt-in by env**: set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`
  (and optionally `LANGFUSE_BASE_URL`) to enable it. Without them the engine
  runs on OpenAI alone and tracing is skipped — no code change needed.
- The CLI flushes buffered traces on exit, so short-lived runs still report.
- **Note:** the LangChain callback captures LLM/chain runs only. Stage 2
  embeddings (`OpenAIEmbeddings`) are **not** emitted as separate Langfuse
  generations — this is inherent to LangChain's callback model.

## Taxonomy governance

- The 16 categories live in [`taxonomy.py`](taxonomy.py) and are the single
  source of truth.
- The classifier always picks from this fixed list; it cannot create new
  categories.
- `add_few_shot_example(category, fingerprint)` lets you reinforce a category
  with new examples at runtime (rejects unknown category names).
# xml_categorization_fingerprint
