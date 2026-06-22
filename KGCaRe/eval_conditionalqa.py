#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kgcare.config import REPO_ROOT
from kgcare.evaluation import evaluate_conditionalqa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a KGCaRe ConditionalQA output.jsonl file.")
    parser.add_argument("--pred-file", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--ref-file", type=Path, default=REPO_ROOT / "data" / "dev.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_file = args.pred_file or (args.run_dir / "output.jsonl" if args.run_dir else None)
    if pred_file is None:
        raise SystemExit("Pass --pred-file or --run-dir.")
    results = evaluate_conditionalqa(pred_file, args.ref_file)
    output_dir = args.run_dir or pred_file.parent
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
