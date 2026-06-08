"""Fixed error taxonomy for the XML error classification engine.

The taxonomy is the single source of truth for Stage 3 classification. New
categories must NEVER be auto-generated at runtime; the classifier must always
pick one of the 16 names defined here (or "Unknown / Unclassified").

Few-shot examples can be appended at runtime via ``add_few_shot_example`` in
``engine.py`` (which mutates the ``examples`` lists below in-memory).
"""

from __future__ import annotations

UNKNOWN_CATEGORY = "Unknown / Unclassified"

TAXONOMY = [
    {
        "name": "Attribute ID Is Missing",
        "description": (
            "A required attribute is missing, undeclared, or not specified in an "
            "XML element, causing the element to fail schema/DTD validation."
        ),
        "examples": [
            {
                "violation_type": "missing_required_attribute",
                "constraint_kind": "id_attribute",
                "scope": "element",
                "actor": "attribute",
            }
        ],
    },
    {
        "name": "Attribute Is Missing",
        "description": (
            "A required XML attribute, element, namespace declaration, or "
            "predefined value is missing, undeclared, or incorrectly specified, "
            "resulting in schema/DTD validation failure."
        ),
        "examples": [
            {
                "violation_type": "missing_required_attribute",
                "constraint_kind": "schema_field",
                "scope": "element",
                "actor": "attribute",
            }
        ],
    },
    {
        "name": "Duplicate ID Attribute",
        "description": (
            "Duplicate identifiers, references, or external IDs are present in the "
            "XML document where unique values are required, causing uniqueness "
            "validation failures."
        ),
        "examples": [
            {
                "violation_type": "uniqueness_violation",
                "constraint_kind": "id_uniqueness",
                "scope": "document",
                "actor": "attribute",
            }
        ],
    },
    {
        "name": "ID Attribute Target Missing",
        "description": (
            "XML cross-references point to identifiers that are missing, undefined, "
            "or have no corresponding target element in the document, resulting in "
            "broken ID/IDREF links."
        ),
        "examples": [
            {
                "violation_type": "broken_reference",
                "constraint_kind": "idref_resolution",
                "scope": "document",
                "actor": "reference",
            }
        ],
    },
    {
        "name": "Invalid Attribute Format",
        "description": (
            "An attribute value does not conform to the required XML data type or "
            "format constraints defined by the schema/DTD."
        ),
        "examples": [
            {
                "violation_type": "format_violation",
                "constraint_kind": "datatype_pattern",
                "scope": "attribute",
                "actor": "value",
            }
        ],
    },
    {
        "name": "Invalid Attribute Value",
        "description": (
            "XML content, metadata, references, or attribute values do not conform "
            "to the required business rules, controlled vocabularies, formatting "
            "standards, or publication specifications."
        ),
        "examples": [
            {
                "violation_type": "invalid_value",
                "constraint_kind": "controlled_vocabulary",
                "scope": "attribute",
                "actor": "value",
            }
        ],
    },
    {
        "name": "Invalid Graphic Asset Declaration",
        "description": (
            "Supplementary, graphic, or multimedia files are missing required "
            "callouts, have unmatched references, or are not properly linked "
            "between the XML and source system records."
        ),
        "examples": [],
    },
    {
        "name": "Invalid ID Attribute Value",
        "description": (
            "ID, IDREF, or IDREFS attribute values do not conform to the required "
            "XML naming conventions and reference syntax, resulting in invalid "
            "identifier format errors."
        ),
        "examples": [],
    },
    {
        "name": "Mandatory Child Element Missing",
        "description": (
            "The XML element is missing one or more required child elements or "
            "content components, resulting in an incomplete structure that does not "
            "satisfy the schema/DTD content model."
        ),
        "examples": [
            {
                "violation_type": "missing_required_child",
                "constraint_kind": "content_model",
                "scope": "structural",
                "actor": "element",
            }
        ],
    },
    {
        "name": "Mismatch In Graphic Asset Declaration",
        "description": (
            "Graphic asset file references declared in the XML do not match the "
            "illustration records, resulting in missing, extra, or inconsistent "
            "asset declarations."
        ),
        "examples": [],
    },
    {
        "name": "Missing End Tag",
        "description": (
            "XML markup contains malformed, incomplete, or improperly closed tags, "
            "resulting in invalid element structure and parsing failures."
        ),
        "examples": [
            {
                "violation_type": "malformed_markup",
                "constraint_kind": "tag_closure",
                "scope": "structural",
                "actor": "element",
            }
        ],
    },
    {
        "name": "Schema Resource Missing",
        "description": (
            "XML parsing failed due to malformed entity references or invalid tag "
            "syntax, preventing the document from being processed correctly."
        ),
        "examples": [],
    },
    {
        "name": "Structural Mismatch Error",
        "description": (
            "The XML element structure or content does not conform to the "
            "schema/DTD-defined content model, resulting in structural mismatch "
            "validation errors."
        ),
        "examples": [
            {
                "violation_type": "element_placement_violation",
                "constraint_kind": "content_model",
                "scope": "structural",
                "actor": "element",
            },
            {
                "violation_type": "type_mismatch",
                "constraint_kind": "node_type",
                "scope": "structural",
                "actor": "node",
            },
        ],
    },
    {
        "name": "Undeclared Unicode Present",
        "description": (
            "The XML document contains unsupported, undeclared, or invalid Unicode "
            "characters that are not permitted by the schema, encoding rules, or "
            "processing system."
        ),
        "examples": [],
    },
    {
        "name": "Undeclared Element",
        "description": (
            "The XML document contains undeclared, unsupported, or disallowed "
            "elements and attributes that are not defined by the schema/DTD or "
            "publication-specific tagging rules."
        ),
        "examples": [
            {
                "violation_type": "undeclared_element",
                "constraint_kind": "schema_definition",
                "scope": "element",
                "actor": "element",
            }
        ],
    },
    {
        "name": "Undeclared Entity Present",
        "description": (
            "The XML document contains invalid, unsupported, undeclared, or "
            "improperly encoded character entities and Unicode characters that "
            "violate parser or encoding validation rules."
        ),
        "examples": [
            {
                "violation_type": "undeclared_entity",
                "constraint_kind": "entity_reference",
                "scope": "document",
                "actor": "entity",
            }
        ],
    },
    {
        "name": UNKNOWN_CATEGORY,
        "description": (
            "Cannot be confidently mapped to any category above. Queued for human "
            "review."
        ),
        "examples": [],
    },
]

# Set of valid category names for fast membership checks / validation.
CATEGORY_NAMES = {entry["name"] for entry in TAXONOMY}


def get_category(name: str) -> dict | None:
    """Return the taxonomy entry for ``name`` or ``None`` if not found."""
    for entry in TAXONOMY:
        if entry["name"] == name:
            return entry
    return None
