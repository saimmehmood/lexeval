"""Command line entry point: `lexeval run configs/mipro.yaml` / `lexeval rerun <run_id>`."""

from __future__ import annotations

import argparse

import pandas as pd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lexeval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run one or more configs")
    p_run.add_argument("configs", nargs="+")
    p_run.add_argument("--experiment", default=None)
    p_run.add_argument("--set", dest="overrides", action="append", default=[],
                       help="override a config value, e.g. --set data.test_size=200")

    p_rerun = sub.add_parser("rerun", help="re-execute a logged MLflow run from its config")
    p_rerun.add_argument("run_id")
    p_rerun.add_argument("--experiment", default=None)

    p_ann = sub.add_parser("summarize-annotations", help="summarize an annotated misclassified.csv")
    p_ann.add_argument("csv", nargs="+")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        from lexeval.config import load_config, with_overrides
        from lexeval.runner import run

        for path in args.configs:
            run(with_overrides(load_config(path), args.overrides), experiment=args.experiment)
    elif args.cmd == "rerun":
        from lexeval.runner import rerun

        rerun(args.run_id, experiment=args.experiment)
    elif args.cmd == "summarize-annotations":
        from lexeval.errors import summarize_annotations

        df = pd.concat([pd.read_csv(p, keep_default_na=False) for p in args.csv])
        print(summarize_annotations(df).to_string(index=False))


if __name__ == "__main__":
    main()
