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

The pipeline is built on **LangChain** (LCEL chains, `langchain-groq`,
`langchain-huggingface`, FAISS). It lives entirely in `engine.py`; `classify.py`
is a thin CLI wrapper (single record + `--input`/`--output` CSV batch mode);
`taxonomy.py` is data.

- **Stage 1 — `normalize_error()`** (LCEL chain: `ChatPromptTemplate | ChatGroq |
  JsonOutputParser | RunnableLambda(_coerce_fingerprint)`, model
  `llama-3.3-70b-versatile`): raw error → an "intent fingerprint" dict with
  exactly the keys in `FINGERPRINT_KEYS` (`violation_type`, `constraint_kind`,
  `scope`, `actor`). Customer-specific values (paths, element/field names, line
  numbers) are stripped by the system prompt so only structural intent remains.
- **Stage 2 — `embed_fingerprint()`** (`HuggingFaceEmbeddings`,
  `all-MiniLM-L6-v2`, 384-dim, L2-normalized): the fingerprint is serialized
  deterministically (`json.dumps(..., sort_keys=True)`) before embedding, so
  identical fingerprints always embed identically. Used only for similarity
  search, **not** for classification.
- **Stage 3 — `classify_error()`** (LCEL chain: `ChatPromptTemplate | ChatGroq |
  StrOutputParser | RunnableLambda(_coerce_category)`): given the fingerprint
  plus the taxonomy definitions and few-shot examples rendered from
  `taxonomy.py`, the LLM must return exactly one category *name*.

System prompts are attached as static `SystemMessage` objects (not template
tuples) so the literal JSON braces in the normalize prompt are not parsed as
LCEL template variables; only the human turn carries a `{variable}`.

`classify_pipeline()` orchestrates all three, assigns a UUID, persists, and
returns the record. `find_similar_errors()` re-normalizes + embeds a query and
queries the FAISS store, converting squared-L2 distance to cosine similarity.

### Key invariant: the taxonomy is closed

`taxonomy.py` is the single source of truth (16 named categories +
`Unknown / Unclassified`). The classifier can **never** invent a category:

- `_coerce_category()` maps the LLM's raw reply to a valid name (exact, then
  case-insensitive) and falls back to `UNKNOWN_CATEGORY` if nothing matches.
- `add_few_shot_example(category, fingerprint)` reinforces an *existing*
  category at runtime by mutating its `examples` list in memory; it raises on
  an unknown category name. The Stage 3 prompt body is rebuilt on every
  `classify_error()` call, so additions take effect immediately — but they are
  **not** persisted across processes.

When changing classification behavior, prefer editing taxonomy `description`s
and `examples` (which feed directly into the Stage 3 prompt) over touching the
prompt-building code.

### Persistence model

`_persist_result()` writes to git-ignored stores, created on first run:

| Store | Contents |
|-------|----------|
| `errors.jsonl` | Human-readable append-only log: full record per line (embedding omitted). |
| `faiss_index/` | LangChain FAISS vector store (`index.faiss` + `index.pkl`); the searchable record of every error, with the full record carried in each document's `metadata`. Backs `find_similar_errors()`. |
| `review_queue.jsonl` | Only the `Unknown / Unclassified` records. |

The FAISS store is the source of truth for similarity search; records are added
pre-embedded via `add_embeddings` (no re-encoding). `errors.jsonl` is a parallel
human-readable log, not used for search. Loading the store uses
`allow_dangerous_deserialization=True` (safe: the index is self-produced).

## Conventions

- LangChain components (`ChatGroq` LLM, `HuggingFaceEmbeddings`, and the two LCEL
  chains) are `@lru_cache`-wrapped lazy singletons so importing `engine` is cheap
  and the torch/groq imports are deferred until first use.
- Retries are LangChain-native: each chain is wrapped with `.with_retry(
  stop_after_attempt=3, wait_exponential_jitter=True)` via the `_retry()` helper.
  `MODEL`, `EMBED_MODEL`, `MAX_TOKENS`, `TEMPERATURE` (0, for determinism) and
  `MAX_RETRIES` are configured at the top of `engine.py`.
- JSON from Stage 1 is parsed by LangChain's `JsonOutputParser`; key presence is
  then guaranteed by `_coerce_fingerprint` (fills any missing `FINGERPRINT_KEYS`
  with `""`).
