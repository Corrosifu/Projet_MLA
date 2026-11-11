"""Command line: ``advrepro reproduce`` runs the study, ``advrepro report`` prints the table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adv_repro.experiments import Config, reproduce
from adv_repro.report import markdown_table


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="advrepro", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("reproduce", help="train, attack and compare with the paper")
    r.add_argument("--dataset", choices=["mnist", "synthetic"], default="mnist")
    r.add_argument("--quick", action="store_true", help="2 epochs per model (smoke run)")
    r.add_argument("--controls", action="store_true", help="also run the two noise controls")
    r.add_argument("--eps", type=float, default=0.25)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--epochs-maxout", type=int, default=None)
    r.add_argument("--epochs-adv", type=int, default=None)
    r.add_argument("--out", default="results")
    r.add_argument("--figures", default="figures")

    t = sub.add_parser("report", help="print the paper-versus-measured table")
    t.add_argument("results", nargs="?", default="results/results.json")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "report":
        print(markdown_table(json.loads(Path(args.results).read_text(encoding="utf-8"))))
        return 0
    overrides = {
        "dataset": args.dataset, "eps": args.eps, "seed": args.seed, "controls": args.controls,
        "out_dir": args.out, "fig_dir": args.figures,
    }  # fmt: skip
    if args.epochs_maxout is not None:
        overrides["epochs_maxout"] = args.epochs_maxout
    if args.epochs_adv is not None:
        overrides["epochs_adv"] = args.epochs_adv
    cfg = Config.quick(**overrides) if args.quick else Config(**overrides)
    results = reproduce(cfg)
    table = markdown_table(results)
    (Path(args.out) / "SUMMARY.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
