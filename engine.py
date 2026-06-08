"""3-Stage XML Error Classification Engine (LangChain implementation).

Pipeline:
    Stage 1  normalize_error()    raw error -> intent fingerprint
             (ChatGroq + ChatPromptTemplate + JsonOutputParser, via LCEL)
    Stage 2  embed_fingerprint()  fingerprint -> embedding vector
             (langchain-huggingface HuggingFaceEmbeddings, all-MiniLM-L6-v2)
    Stage 3  classify_error()     fingerprint -> one canonical category
             (ChatGroq + ChatPromptTemplate + StrOutputParser, via LCEL)

Every classified error is stored in a LangChain FAISS vector store, which backs
``find_similar_errors()``. The orchestrator ``classify_pipeline()`` runs all
three stages, persists the result, and returns the record.
"""

from __future__ import annotations

import json
import os
import uuid
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

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
REVIEW_QUEUE_FILE = BASE_DIR / "review_queue.jsonl"
FAISS_DIR = BASE_DIR / "faiss_index"

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

# SystemMessage (not a template tuple) so the literal JSON braces in the
# normalize prompt are not parsed as template variables. The human turn carries
# the only variable; substituted values are inserted literally, never re-parsed.
NORMALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(content=NORMALIZE_SYSTEM_PROMPT), ("human", "{raw_error}")]
)
CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(content=CLASSIFY_SYSTEM_PROMPT), ("human", "{user_message}")]
)


# --------------------------------------------------------------------------- #
# Lazily-initialised LangChain components
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _llm():
    from langchain_groq import ChatGroq

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatGroq(model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)


@lru_cache(maxsize=1)
def _embedder():
    # Imported here so the heavy torch / sentence-transformers import only
    # happens when embeddings are actually needed.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def _retry(runnable):
    """Wrap a runnable with 3 attempts and exponential backoff (LangChain)."""
    return runnable.with_retry(
        stop_after_attempt=MAX_RETRIES,
        wait_exponential_jitter=True,
    )


@lru_cache(maxsize=1)
def _normalize_chain():
    # raw_error -> JSON -> fingerprint with exactly FINGERPRINT_KEYS.
    chain = (
        NORMALIZE_PROMPT
        | _llm()
        | JsonOutputParser()
        | RunnableLambda(_coerce_fingerprint)
    )
    return _retry(chain)


@lru_cache(maxsize=1)
def _classify_chain():
    # user_message -> raw category string -> validated taxonomy name.
    chain = (
        CLASSIFY_PROMPT
        | _llm()
        | StrOutputParser()
        | RunnableLambda(_coerce_category)
    )
    return _retry(chain)


# --------------------------------------------------------------------------- #
# Stage 1 — Normalize
# --------------------------------------------------------------------------- #

def _coerce_fingerprint(parsed: dict) -> dict:
    """Guarantee all expected keys exist (empty string if the LLM omitted one)."""
    parsed = parsed or {}
    return {key: parsed.get(key, "") for key in FINGERPRINT_KEYS}


def normalize_error(raw_error: str) -> dict:
    """Extract a structured intent fingerprint from a raw error message."""
    return _normalize_chain().invoke({"raw_error": raw_error})


# --------------------------------------------------------------------------- #
# Stage 2 — Embed
# --------------------------------------------------------------------------- #

def _stable_fingerprint_string(fingerprint: dict) -> str:
    """Serialize a fingerprint to a deterministic string (sorted keys)."""
    return json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))


def embed_fingerprint(fingerprint: dict) -> list[float]:
    """Generate an embedding vector for a fingerprint dict."""
    text = _stable_fingerprint_string(fingerprint)
    return _embedder().embed_query(text)


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
    return _classify_chain().invoke({"user_message": user_message})


def _coerce_category(raw: str) -> str:
    """Map a raw LLM response to a valid taxonomy name.

    Never invents a category: anything that does not match the fixed list
    becomes ``Unknown / Unclassified``.
    """
    candidate = (raw or "").strip().strip(".").strip('"').strip()

    if candidate in CATEGORY_NAMES:
        return candidate

    lowered = candidate.lower()
    for name in CATEGORY_NAMES:
        if name.lower() == lowered:
            return name

    return UNKNOWN_CATEGORY


# --------------------------------------------------------------------------- #
# Taxonomy governance
# --------------------------------------------------------------------------- #

def add_few_shot_example(category: str, fingerprint: dict) -> None:
    """Append a few-shot example to an existing taxonomy category at runtime.

    The example is read fresh into the prompt on each ``classify_error`` call,
    so additions take effect immediately (the chain is cached, the prompt body
    is rebuilt per call).
    """
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
# Vector store (FAISS) + storage
# --------------------------------------------------------------------------- #

def _load_vectorstore():
    """Load the persisted FAISS store, or return None if none exists yet."""
    from langchain_community.vectorstores import FAISS

    if not FAISS_DIR.exists():
        return None
    # Local, self-produced index -> safe to deserialize.
    return FAISS.load_local(
        str(FAISS_DIR), _embedder(), allow_dangerous_deserialization=True
    )


def _add_to_vectorstore(text: str, embedding: list[float], metadata: dict) -> None:
    """Add one pre-embedded record to the FAISS store and persist it."""
    from langchain_community.vectorstores import FAISS

    vs = _load_vectorstore()
    if vs is None:
        vs = FAISS.from_embeddings(
            [(text, embedding)], _embedder(), metadatas=[metadata]
        )
    else:
        vs.add_embeddings([(text, embedding)], metadatas=[metadata])
    vs.save_local(str(FAISS_DIR))


def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _persist_result(result: dict) -> None:
    """Write a classified error to the log files and the FAISS store."""
    record = {k: v for k, v in result.items() if k != "embedding"}

    # Human-readable append-only log of every classification.
    _append_jsonl(ERRORS_FILE, record)

    # Searchable store: the fingerprint text + its precomputed embedding, with
    # the full record carried as metadata (used by find_similar_errors).
    _add_to_vectorstore(
        _stable_fingerprint_string(result["fingerprint"]),
        result["embedding"],
        record,
    )

    # Unknown classifications are queued for human review.
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

def _l2_to_cosine(squared_l2: float) -> float:
    """Convert FAISS squared-L2 distance to cosine similarity.

    Embeddings are L2-normalized, so ||a - b||^2 = 2 - 2*cos, giving
    cos = 1 - dist/2 (higher = more similar).
    """
    return float(1.0 - squared_l2 / 2.0)


def find_similar_errors(raw_error: str, top_k: int = 5) -> list[dict]:
    """Return the top-k most similar previously-classified errors.

    Normalizes and embeds the input, then queries the FAISS vector store.
    """
    vs = _load_vectorstore()
    if vs is None:
        return []

    fingerprint = normalize_error(raw_error)
    vector = embed_fingerprint(fingerprint)
    hits = vs.similarity_search_with_score_by_vector(vector, k=top_k)

    results = []
    for doc, score in hits:
        md = doc.metadata
        results.append(
            {
                "id": md.get("id"),
                "customer_id": md.get("customer_id"),
                "raw_error": md.get("raw_error"),
                "fingerprint": md.get("fingerprint"),
                "category": md.get("category"),
                "similarity": _l2_to_cosine(score),
            }
        )
    return results
