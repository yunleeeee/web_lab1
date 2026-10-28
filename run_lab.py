"""Entry point for the Qilin-IR-50K course experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 1: Chinese information retrieval")
    parser.add_argument("stage", choices=["prepare", "lexical", "semantic", "report", "all", "test"])
    parser.add_argument("--data", type=Path, default=Path("Qilin-IR-50K"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--model-path", type=Path, default=Path("models/bge-small-zh-v1.5"))
    parser.add_argument("--skip-semantic", action="store_true", help="只运行传统检索；报告会标明模型部分缺失")
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        from lab_ir.preprocessing import prepare
        prepare(args.data, args.output)
    if args.stage in ("lexical", "all"):
        from lab_ir.lexical_experiments import run_experiments
        run_experiments(args.data, args.output)
    if args.stage in ("semantic", "all") and not args.skip_semantic:
        command = [sys.executable, "-m", "lab_ir.semantic_experiments", "--data", str(args.data),
                   "--output", str(args.output), "--model-path", str(args.model_path)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
