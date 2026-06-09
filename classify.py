"""CLI entry point for the XML error classification engine.

Two modes (both reuse the same ``engine.classify_pipeline`` per record):

    Single record:
        python classify.py --error "your error here" --customer "customer_a"
        python classify.py --error "broken idref" --customer "customer_b" --similar

    Batch (CSV in, CSV out):
        python classify.py --input errors.csv --output classified_errors.csv

The input CSV must have an ``error_message`` column and a ``customer`` column.
The output CSV preserves all original columns and appends ``category`` and
``confidence`` (``confidence`` is blank when the pipeline does not provide one).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from engine import classify_pipeline, find_similar_errors

# Input CSV columns the batch mode reads from, and the columns it appends.
ERROR_COLUMN = "error_message"
CUSTOMER_COLUMN = "customer"
APPENDED_COLUMNS = ("category", "confidence")


def run_single(args: argparse.Namespace) -> int:
    """Classify one error (existing behavior — unchanged)."""
    try:
        result = classify_pipeline(args.error, args.customer)
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Keep the printed output readable: drop the raw embedding vector, show its size.
    printable = {k: v for k, v in result.items() if k != "embedding"}
    printable["embedding_dim"] = len(result["embedding"])
    print(json.dumps(printable, indent=2, ensure_ascii=False))

    if args.similar:
        # find_similar_errors includes the just-stored error; that's expected.
        similar = find_similar_errors(args.error, top_k=args.top_k)
        print("\nTop similar past errors:")
        print(json.dumps(similar, indent=2, ensure_ascii=False))

    return 0


def run_batch(input_path: str, output_path: str) -> int:
    """Classify every row of an input CSV and write an output CSV.

    Reuses ``classify_pipeline`` per row. Rows that are missing the error
    message / customer, or that fail to classify, are skipped and logged to
    stderr without stopping the run.
    """
    try:
        infile = open(input_path, newline="", encoding="utf-8")
    except OSError as exc:
        print(f"Error: cannot open input file {input_path!r}: {exc}", file=sys.stderr)
        return 1

    with infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []

        missing = [c for c in (ERROR_COLUMN, CUSTOMER_COLUMN) if c not in fieldnames]
        if missing:
            print(
                f"Error: input CSV is missing required column(s): {', '.join(missing)}. "
                f"Expected columns include {ERROR_COLUMN!r} and {CUSTOMER_COLUMN!r}.",
                file=sys.stderr,
            )
            return 1

        # Output columns: all original columns + the appended ones (no duplicates).
        out_fieldnames = list(fieldnames) + [
            c for c in APPENDED_COLUMNS if c not in fieldnames
        ]

        output_rows = []
        processed = 0
        skipped = 0

        # start=2: row 1 is the header, so the first data row is line 2.
        for line_no, row in enumerate(reader, start=2):
            error_message = (row.get(ERROR_COLUMN) or "").strip()
            customer = (row.get(CUSTOMER_COLUMN) or "").strip()

            if not error_message or not customer:
                skipped += 1
                print(
                    f"Skipping row {line_no}: missing "
                    f"{ERROR_COLUMN!r} or {CUSTOMER_COLUMN!r}.",
                    file=sys.stderr,
                )
                continue

            try:
                result = classify_pipeline(error_message, customer)
            except Exception as exc:  # noqa: BLE001 - skip and keep going
                skipped += 1
                print(
                    f"Skipping row {line_no} (customer={customer!r}): "
                    f"classification failed: {exc}",
                    file=sys.stderr,
                )
                continue

            out_row = dict(row)
            out_row["category"] = result["category"]
            # The pipeline does not currently emit a confidence score; stay
            # honest and leave it blank unless one is ever provided.
            out_row["confidence"] = result.get("confidence", "")
            output_rows.append(out_row)
            processed += 1

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=out_fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
    except OSError as exc:
        print(f"Error: cannot write output file {output_path!r}: {exc}", file=sys.stderr)
        return 1

    print(
        f"Done. Classified {processed} row(s), skipped {skipped}. "
        f"Output written to {output_path}."
    )
    # Non-zero exit only if nothing was classified at all.
    return 0 if processed > 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify raw XML errors into the canonical taxonomy "
        "(single record or batch CSV)."
    )
    parser.add_argument("--error", help="Raw error message to classify (single mode).")
    parser.add_argument(
        "--customer", help="Customer identifier, e.g. customer_a (single mode)."
    )
    parser.add_argument(
        "--input", help="Path to an input CSV for batch mode "
        f"(requires columns {ERROR_COLUMN!r} and {CUSTOMER_COLUMN!r})."
    )
    parser.add_argument(
        "--output", help="Path to write the classified output CSV (batch mode)."
    )
    parser.add_argument(
        "--similar",
        action="store_true",
        help="Single mode only: also show the top-k most similar past errors.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of similar errors to show."
    )
    args = parser.parse_args(argv)

    try:
        # Batch mode takes precedence when --input is given.
        if args.input:
            if not args.output:
                parser.error("--input requires --output (the CSV to write results to).")
            return run_batch(args.input, args.output)

        # Otherwise fall back to the existing single-record mode.
        if args.error and args.customer:
            return run_single(args)

        parser.error(
            "provide either --error and --customer (single mode), "
            "or --input and --output (batch CSV mode)."
        )
    finally:
        _flush_langfuse()


def _flush_langfuse() -> None:
    """Flush buffered Langfuse traces before this short-lived process exits.

    No-op (and silent) when Langfuse isn't configured.
    """
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:  # noqa: BLE001 - tracing must never break the CLI
        pass


if __name__ == "__main__":
    raise SystemExit(main())
