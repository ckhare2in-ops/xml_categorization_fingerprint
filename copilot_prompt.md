# Copilot Prompt — 3-Stage XML Error Classification Engine

Build a 3-stage error classification engine in Python that normalizes raw error messages from multiple customers into a consistent taxonomy.

---

## Architecture Overview

```
Customer Raw Error
       │
       ▼
Stage 1 — Normalize (LLM)
  Strip customer-specific values → extract intent fingerprint
       │
       ▼
Stage 2 — Embed (sentence-transformers)
  Fingerprint → embedding vector for similarity search
       │
       ▼
Stage 3 — Classify (LLM + fixed taxonomy)
  Pick exactly one of 16 categories
       │
       ▼
Canonical Category (consistent across all customers)
```

---

## Stage 1 — Normalize (LLM Extraction)

Build a function:

```python
def normalize_error(raw_error: str) -> dict
```

Call the LLM using the **Groq SDK** (`groq` Python package) with the following system prompt:

**SYSTEM:**
```
You are an error intent extractor for a multi-product XML error classification engine.
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

No explanation. No markdown. Raw JSON only.
```

**USER:** `{raw_error}`

---

## Stage 2 — Embed Normalized Fingerprint

Build a function:

```python
def embed_fingerprint(fingerprint: dict) -> list[float]
```

- Serialize the fingerprint dict to a stable string (sorted keys)
- Generate an embedding using `sentence-transformers` model `all-MiniLM-L6-v2`
- Return the embedding vector

---

## Stage 3 — Classify Against Fixed Taxonomy

Build a function:

```python
def classify_error(fingerprint: dict) -> str
```

Call the LLM using the **Groq SDK** with the following system prompt:

**SYSTEM:**
```
You are an XML error classifier for a publishing pipeline.
You will receive a structured error fingerprint and must assign it to EXACTLY ONE category from the taxonomy below.
Return ONLY the exact category name — no explanation, no punctuation, nothing else.
Only return "Unknown / Unclassified" if the fingerprint genuinely does not fit any category.
```

### Taxonomy (16 Categories)

| Category | Definition |
|----------|------------|
| **Attribute ID Is Missing** | A required attribute is missing, undeclared, or not specified in an XML element, causing the element to fail schema/DTD validation. |
| **Attribute Is Missing** | A required XML attribute, element, namespace declaration, or predefined value is missing, undeclared, or incorrectly specified, resulting in schema/DTD validation failure. |
| **Duplicate ID Attribute** | Duplicate identifiers, references, or external IDs are present in the XML document where unique values are required, causing uniqueness validation failures. |
| **ID Attribute Target Missing** | XML cross-references point to identifiers that are missing, undefined, or have no corresponding target element in the document, resulting in broken ID/IDREF links. |
| **Invalid Attribute Format** | An attribute value does not conform to the required XML data type or format constraints defined by the schema/DTD. |
| **Invalid Attribute Value** | XML content, metadata, references, or attribute values do not conform to the required business rules, controlled vocabularies, formatting standards, or publication specifications. |
| **Invalid Graphic Asset Declaration** | Supplementary, graphic, or multimedia files are missing required callouts, have unmatched references, or are not properly linked between the XML and source system records. |
| **Invalid ID Attribute Value** | ID, IDREF, or IDREFS attribute values do not conform to the required XML naming conventions and reference syntax, resulting in invalid identifier format errors. |
| **Mandatory Child Element Missing** | The XML element is missing one or more required child elements or content components, resulting in an incomplete structure that does not satisfy the schema/DTD content model. |
| **Mismatch In Graphic Asset Declaration** | Graphic asset file references declared in the XML do not match the illustration records, resulting in missing, extra, or inconsistent asset declarations. |
| **Missing End Tag** | XML markup contains malformed, incomplete, or improperly closed tags, resulting in invalid element structure and parsing failures. |
| **Schema Resource Missing** | XML parsing failed due to malformed entity references or invalid tag syntax, preventing the document from being processed correctly. |
| **Structural Mismatch Error** | The XML element structure or content does not conform to the schema/DTD-defined content model, resulting in structural mismatch validation errors. |
| **Undeclared Unicode Present** | The XML document contains unsupported, undeclared, or invalid Unicode characters that are not permitted by the schema, encoding rules, or processing system. |
| **Undeclared Element** | The XML document contains undeclared, unsupported, or disallowed elements and attributes that are not defined by the schema/DTD or publication-specific tagging rules. |
| **Undeclared Entity Present** | The XML document contains invalid, unsupported, undeclared, or improperly encoded character entities and Unicode characters that violate parser or encoding validation rules. |
| **Unknown / Unclassified** | Cannot be confidently mapped to any category above. Queued for human review. |

### Few-Shot Examples (include in the USER message)

```
- {"violation_type": "element_placement_violation", "constraint_kind": "content_model", "scope": "structural", "actor": "element"} → Structural Mismatch Error
- {"violation_type": "type_mismatch", "constraint_kind": "node_type", "scope": "structural", "actor": "node"} → Structural Mismatch Error
- {"violation_type": "missing_required_child", "constraint_kind": "content_model", "scope": "structural", "actor": "element"} → Mandatory Child Element Missing
- {"violation_type": "missing_required_attribute", "constraint_kind": "id_attribute", "scope": "element", "actor": "attribute"} → Attribute ID Is Missing
- {"violation_type": "missing_required_attribute", "constraint_kind": "schema_field", "scope": "element", "actor": "attribute"} → Attribute Is Missing
- {"violation_type": "uniqueness_violation", "constraint_kind": "id_uniqueness", "scope": "document", "actor": "attribute"} → Duplicate ID Attribute
- {"violation_type": "broken_reference", "constraint_kind": "idref_resolution", "scope": "document", "actor": "reference"} → ID Attribute Target Missing
- {"violation_type": "format_violation", "constraint_kind": "datatype_pattern", "scope": "attribute", "actor": "value"} → Invalid Attribute Format
- {"violation_type": "invalid_value", "constraint_kind": "controlled_vocabulary", "scope": "attribute", "actor": "value"} → Invalid Attribute Value
- {"violation_type": "malformed_markup", "constraint_kind": "tag_closure", "scope": "structural", "actor": "element"} → Missing End Tag
- {"violation_type": "undeclared_element", "constraint_kind": "schema_definition", "scope": "element", "actor": "element"} → Undeclared Element
- {"violation_type": "undeclared_entity", "constraint_kind": "entity_reference", "scope": "document", "actor": "entity"} → Undeclared Entity Present
```

**USER:** `fingerprint: {fingerprint_json}`

---

## Orchestrator

Build a function:

```python
def classify_pipeline(raw_error: str, customer_id: str) -> dict
```

Steps:
1. Call `normalize_error()` → fingerprint
2. Call `embed_fingerprint()` → vector (store for future similarity search)
3. Call `classify_error()` → category
4. Return:

```python
{
  "customer_id": customer_id,
  "raw_error": raw_error,
  "fingerprint": fingerprint,
  "category": category,
  "embedding": vector
}
```

---

## Taxonomy Governance

- Store the taxonomy as a Python list of dicts:
  ```python
  [{ "name": "...", "description": "...", "examples": [...] }]
  ```
  Use the 16 categories and definitions from the table above.
- **Never auto-generate new categories** — always pick from the fixed list.
- If `classify_error()` returns `"Unknown / Unclassified"`, log it to `review_queue.jsonl`.
- Add a function:
  ```python
  def add_few_shot_example(category: str, fingerprint: dict)
  ```
  to update examples at runtime.

---

## Storage & Feedback Loop

- Save each result to `errors.jsonl` (one JSON object per line)
- Save embeddings to a numpy `.npy` file indexed by a UUID per error
- Add a function:
  ```python
  def find_similar_errors(raw_error: str, top_k: int = 5) -> list[dict]
  ```
  Steps:
  1. Normalize and embed the input
  2. Compute cosine similarity against stored embeddings
  3. Return top-k most similar past errors with their categories

---

## Setup Requirements

| Requirement | Detail |
|-------------|--------|
| **Libraries** | `groq`, `sentence-transformers`, `numpy`, `python-dotenv` |
| **API Key** | Load `GROQ_API_KEY` from `.env` |
| **Model** | `llama-3.3-70b-versatile` |
| **LLM settings** | `max_tokens=200`, `temperature=0` |
| **Retry logic** | Max 3 attempts per LLM call, exponential backoff |
| **CLI** | `python classify.py --error "your error here" --customer "customer_a"` |
| **README** | Include setup instructions + example outputs for at least 2 different customer error formats that both map to `Structural Mismatch Error` |

---

## Output Files

| File | Purpose |
|------|---------|
| `errors.jsonl` | All classified errors, one JSON per line |
| `embeddings.npy` | Numpy array of all embedding vectors |
| `embeddings_meta.jsonl` | Metadata index per embedding (id, customer, category) |
| `review_queue.jsonl` | Errors that returned `Unknown / Unclassified` |
