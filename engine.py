"""3-Stage XML Error Classification Engine.

Pipeline:
    Stage 1  normalize_error()    raw error -> intent fingerprint (Groq LLM)
    Stage 2  embed_fingerprint()  fingerprint -> embedding vector (MiniLM)
    Stage 3  classify_error()     fingerprint -> one canonical category (Groq LLM)

The orchestrator ``classify_pipeline()`` runs all three stages, persists the
result, and stores the embedding for later similarity search.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq

from taxonomy import CATEGORY_NAMES, TAXONOMY, UNKNOWN_CATEGORY, get_category

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_TOKENS = 200
TEMPERATURE = 0
MAX_RETRIES = 3

BASE_DIR = Path(__file__).resolve().parent
ERRORS_FILE = BASE_DIR / "errors.jsonl"
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npy"
EMBEDDINGS_META_FILE = BASE_DIR / "embeddings_meta.jsonl"
REVIEW_QUEUE_FILE = BASE_DIR / "review_queue.jsonl"

FINGERPRINT_KEYS = ("violation_type", "constraint_kind", "scope", "actor")

NORMALIZE_SYSTEM_PROMPT = """You are an error intent extractor for a multi-product XML error classification engine.
Given a raw error message, extract a structured "intent fingerprint" in JSON.
Strip all customer-specific values: file paths, field names, element names, type names, line numbers, node IDs.
Focus only on the structural intent — what KIND of violation is this?

Return ONLY valid JSON with these keys:
{
  "violation_type": "",
  "constraint_kind": "",
  "scope": "",
  "actor": ""
}

No explanation. No markdown. Raw JSON only."""

CLASSIFY_SYSTEM_PROMPT = """You are an XML error classifier for a publishing pipeline.
You will receive a structured error fingerprint and must assign it to EXACTLY ONE category from the taxonomy below.
Return ONLY the exact category name — no explanation, no punctuation, nothing else.
Only return "Unknown / Unclassified" if the fingerprint genuinely does not fit any category."""


# --------------------------------------------------------------------------- #
# Lazily-initialised clients (so importing the module is cheap / test-friendly)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Groq(api_key=api_key)


@lru_cache(maxsize=1)
def _embedder():
    # Imported here so the (heavy) torch/sentence-transformers import only
    # happens when embeddings are actually needed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #

def _with_retries(fn, *, what: str):
    """Call ``fn`` with up to MAX_RETRIES attempts and exponential backoff."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise after retries
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait)
    raise RuntimeError(f"{what} failed after {MAX_RETRIES} attempts") from last_exc


# --------------------------------------------------------------------------- #
# Stage 1 — Normalize
# --------------------------------------------------------------------------- #

def normalize_error(raw_error: str) -> dict:
    """Extract a structured intent fingerprint from a raw error message."""

    def _call():
        resp = _groq_client().chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": NORMALIZE_SYSTEM_PROMPT},
                {"role": "user", "content": raw_error},
            ],
        )
        return resp.choices[0].message.content

    content = _with_retries(_call, what="normalize_error")
    fingerprint = _parse_json(content)

    # Guarantee all expected keys exist (empty string if the LLM omitted one).
    return {key: fingerprint.get(key, "") for key in FINGERPRINT_KEYS}


# --------------------------------------------------------------------------- #
# Stage 2 — Embed
# --------------------------------------------------------------------------- #

def _stable_fingerprint_string(fingerprint: dict) -> str:
    """Serialize a fingerprint to a deterministic string (sorted keys)."""
    return json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))


def embed_fingerprint(fingerprint: dict) -> list[float]:
    """Generate an embedding vector for a fingerprint dict."""
    text = _stable_fingerprint_string(fingerprint)
    vector = _embedder().encode(text, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32).tolist()


# --------------------------------------------------------------------------- #
# Stage 3 — Classify
# --------------------------------------------------------------------------- #

def _build_few_shot_block() -> str:
    """Render the current taxonomy examples as the few-shot reference block."""
    lines = []
    for entry in TAXONOMY:
        for example in entry["examples"]:
            lines.append(
                f"- {json.dumps(example, sort_keys=True)} → {entry['name']}"
            )
    return "\n".join(lines)


def _build_taxonomy_block() -> str:
    """Render the taxonomy names + definitions for the classifier prompt."""
    return "\n".join(
        f"- {entry['name']}: {entry['description']}" for entry in TAXONOMY
    )


def classify_error(fingerprint: dict) -> str:
    """Assign the fingerprint to exactly one canonical taxonomy category."""
    fingerprint_json = _stable_fingerprint_string(fingerprint)

    user_message = (
        "Taxonomy categories and definitions:\n"
        f"{_build_taxonomy_block()}\n\n"
        "Few-shot examples:\n"
        f"{_build_few_shot_block()}\n\n"
        f"fingerprint: {fingerprint_json}"
    )

    def _call():
        resp = _groq_client().chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content

    raw = _with_retries(_call, what="classify_error")
    return _coerce_category(raw)


def _coerce_category(raw: str) -> str:
    """Map a raw LLM response to a valid taxonomy name.

    Never invents a category: anything that does not match the fixed list
    becomes ``Unknown / Unclassified``.
    """
    candidate = (raw or "").strip().strip(".").strip('"').strip()

    if candidate in CATEGORY_NAMES:
        return candidate

    # Case-insensitive fallback match.
    lowered = candidate.lower()
    for name in CATEGORY_NAMES:
        if name.lower() == lowered:
            return name

    return UNKNOWN_CATEGORY


# --------------------------------------------------------------------------- #
# Taxonomy governance
# --------------------------------------------------------------------------- #

def add_few_shot_example(category: str, fingerprint: dict) -> None:
    """Append a few-shot example to an existing taxonomy category at runtime."""
    entry = get_category(category)
    if entry is None:
        raise ValueError(
            f"Unknown category {category!r}; refusing to create a new category. "
            f"Valid categories: {sorted(CATEGORY_NAMES)}"
        )
    cleaned = {key: fingerprint.get(key, "") for key in FINGERPRINT_KEYS}
    if cleaned not in entry["examples"]:
        entry["examples"].append(cleaned)


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _store_embedding(vector: list[float]) -> None:
    """Append a vector as a new row in the embeddings.npy matrix."""
    arr = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    if EMBEDDINGS_FILE.exists():
        existing = np.load(EMBEDDINGS_FILE)
        if existing.size == 0:
            stacked = arr
        else:
            stacked = np.vstack([existing, arr])
    else:
        stacked = arr
    np.save(EMBEDDINGS_FILE, stacked)


def _persist_result(result: dict) -> None:
    """Write a classified error to all output files."""
    # 1. Full record -> errors.jsonl (without the bulky embedding inline).
    record = {k: v for k, v in result.items() if k != "embedding"}
    _append_jsonl(ERRORS_FILE, record)

    # 2. Embedding vector -> embeddings.npy (row order matches meta file).
    _store_embedding(result["embedding"])

    # 3. Embedding metadata index -> embeddings_meta.jsonl.
    _append_jsonl(
        EMBEDDINGS_META_FILE,
        {
            "id": result["id"],
            "customer_id": result["customer_id"],
            "category": result["category"],
        },
    )

    # 4. Unknown classifications -> review_queue.jsonl.
    if result["category"] == UNKNOWN_CATEGORY:
        _append_jsonl(REVIEW_QUEUE_FILE, record)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def classify_pipeline(raw_error: str, customer_id: str) -> dict:
    """Run all three stages, persist the result, and return it."""
    fingerprint = normalize_error(raw_error)
    vector = embed_fingerprint(fingerprint)
    category = classify_error(fingerprint)

    result = {
        "id": str(uuid.uuid4()),
        "customer_id": customer_id,
        "raw_error": raw_error,
        "fingerprint": fingerprint,
        "category": category,
        "embedding": vector,
    }

    _persist_result(result)
    return result


# --------------------------------------------------------------------------- #
# Similarity search (feedback loop)
# --------------------------------------------------------------------------- #

def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and each row of a matrix."""
    q_norm = query / (np.linalg.norm(query) + 1e-12)
    m_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return m_norms @ q_norm


def _load_errors() -> list[dict]:
    if not ERRORS_FILE.exists():
        return []
    with ERRORS_FILE.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def find_similar_errors(raw_error: str, top_k: int = 5) -> list[dict]:
    """Return the top-k most similar previously-classified errors.

    Normalizes and embeds the input, then ranks stored embeddings by cosine
    similarity. Row order in ``embeddings.npy`` matches line order in
    ``errors.jsonl`` / ``embeddings_meta.jsonl``.
    """
    if not EMBEDDINGS_FILE.exists():
        return []

    matrix = np.load(EMBEDDINGS_FILE)
    if matrix.size == 0:
        return []

    errors = _load_errors()
    fingerprint = normalize_error(raw_error)
    query = np.asarray(embed_fingerprint(fingerprint), dtype=np.float32)

    scores = _cosine_similarity(query, matrix)
    n = min(top_k, len(scores), len(errors))
    top_idx = np.argsort(scores)[::-1][:n]

    results = []
    for idx in top_idx:
        record = errors[idx]
        results.append(
            {
                "id": record.get("id"),
                "customer_id": record.get("customer_id"),
                "raw_error": record.get("raw_error"),
                "fingerprint": record.get("fingerprint"),
                "category": record.get("category"),
                "similarity": float(scores[idx]),
            }
        )
    return results


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _parse_json(content: str) -> dict:
    """Parse JSON from an LLM response, tolerating markdown code fences."""
    text = (content or "").strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present.
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: extract the first {...} block.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
