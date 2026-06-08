# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment (Python 3.9+). Re-activate in each new shell.
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure the Groq key (required for any LLM call)
cp .env.example .env   # then set GROQ_API_KEY

# Run the classifier
python classify.py --error "your error here" --customer "customer_a"
python classify.py --error "..." --customer "customer_b" --similar --top-k 3
```

There is no test suite, linter config, or build step. The `.venv/` is the
supported, git-ignored toolchain (it isolates the heavy `torch` /
`sentence-transformers` deps); always activate it before running.

## Architecture

A 3-stage pipeline that normalizes raw XML error messages from *different
customers* (who phrase the "same" problem differently) into one shared,
fixed taxonomy. The whole point is convergence: distinct wordings must land on
the same canonical category.

The pipeline lives entirely in `engine.py`; `classify.py` is a thin CLI wrapper;
`taxonomy.py` is data.

- **Stage 1 — `normalize_error()`** (Groq LLM, `llama-3.3-70b-versatile`):
  raw error → an "intent fingerprint" dict with exactly the keys in
  `FINGERPRINT_KEYS` (`violation_type`, `constraint_kind`, `scope`, `actor`).
  Customer-specific values (paths, element/field names, line numbers) are
  stripped by the system prompt so only structural intent remains.
- **Stage 2 — `embed_fingerprint()`** (`all-MiniLM-L6-v2`, 384-dim): the
  fingerprint is serialized deterministically (`json.dumps(..., sort_keys=True)`)
  before embedding, so identical fingerprints always embed identically. Used
  only for similarity search, **not** for classification.
- **Stage 3 — `classify_error()`** (Groq LLM): given the fingerprint plus the
  taxonomy definitions and few-shot examples rendered from `taxonomy.py`, the
  LLM must return exactly one category *name*.

`classify_pipeline()` orchestrates all three, assigns a UUID, persists, and
returns the record. `find_similar_errors()` re-normalizes + embeds a query and
ranks stored embeddings by cosine similarity.

### Key invariant: the taxonomy is closed

`taxonomy.py` is the single source of truth (16 named categories +
`Unknown / Unclassified`). The classifier can **never** invent a category:

- `_coerce_category()` maps the LLM's raw reply to a valid name (exact, then
  case-insensitive) and falls back to `UNKNOWN_CATEGORY` if nothing matches.
- `add_few_shot_example(category, fingerprint)` reinforces an *existing*
  category at runtime by mutating its `examples` list in memory; it raises on
  an unknown category name. These additions are **not** persisted across
  processes.

When changing classification behavior, prefer editing taxonomy `description`s
and `examples` (which feed directly into the Stage 3 prompt) over touching the
prompt-building code.

### Persistence model (positional coupling)

`_persist_result()` appends to four git-ignored files, created on first run.
Their row/line ordering is a load-bearing contract — `find_similar_errors()`
relies on `embeddings.npy` row *i* corresponding to line *i* of `errors.jsonl`
and `embeddings_meta.jsonl`:

| File | Contents |
|------|----------|
| `errors.jsonl` | Full record per line (embedding omitted). |
| `embeddings.npy` | Stacked 384-dim float32 matrix, one row per error. |
| `embeddings_meta.jsonl` | `id` / `customer_id` / `category` index. |
| `review_queue.jsonl` | Only the `Unknown / Unclassified` records. |

Never reorder or filter one of these files without doing the same to the others,
or similarity results will silently misattribute.

## Conventions

- LLM/embedder clients are `@lru_cache`-wrapped lazy singletons so importing
  `engine` is cheap and the torch import is deferred until embeddings are needed.
- Every LLM call goes through `_with_retries()` (3 attempts, 1s/2s/4s backoff).
  `MODEL`, `EMBED_MODEL`, `MAX_TOKENS`, `TEMPERATURE` (0, for determinism) are
  configured at the top of `engine.py`.
- LLM JSON is parsed via `_parse_json()`, which tolerates markdown code fences
  and falls back to extracting the first `{...}` block.
