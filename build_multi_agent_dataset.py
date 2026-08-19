#!/usr/bin/env python3
"""Build the 2,304-row multi-agent safety-evaluation JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from tools.dataset_v2 import (
    DEFAULT_OUTPUT_JSONL,
    DEFAULT_SOURCE_XLSX,
    build_manifest_rows,
    jsonl_sha256,
    write_jsonl,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(rows: Sequence[dict], output: Path | None) -> dict:
    summary = {
        "rows": len(rows),
        "semantic_cases": len({row["case_id"] for row in rows}),
        "intent_pairs": len({row["intent_pair_id"] for row in rows}),
        "replicates": sorted({row["replicate"] for row in rows}),
        "request_types": dict(sorted(Counter(row["request_type"] for row in rows).items())),
        "domains": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "scenarios": dict(sorted(Counter(row["scenario"] for row in rows).items())),
        "system_conditions": dict(sorted(Counter(row["system_condition"] for row in rows).items())),
        "modes": dict(sorted(Counter(row["mode"] for row in rows).items())),
    }
    if output is not None:
        summary["output"] = str(output)
        summary["output_sha256"] = jsonl_sha256(output)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_XLSX,
        help=f"Read-only source XLSX (default: {DEFAULT_SOURCE_XLSX})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT_JSONL})",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build and validate in memory without writing JSONL.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_digest_before = _file_sha256(source)

    rows = build_manifest_rows(source)
    output: Path | None = None
    if not args.validate_only:
        output = write_jsonl(rows, args.output.resolve())

    source_digest_after = _file_sha256(source)
    if source_digest_after != source_digest_before:
        raise RuntimeError("The read-only source XLSX changed while building the manifest.")

    report = _summary(rows, output)
    report["source"] = str(source)
    report["source_sha256"] = source_digest_before
    report["status"] = "validated" if args.validate_only else "written"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
