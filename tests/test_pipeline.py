import json
import math

import pytest

from adv_repro.cli import build_parser, main
from adv_repro.experiments import Config, reproduce
from adv_repro.paper import PAPER
from adv_repro.report import LABELS, MEASURED, comparison_rows, markdown_table


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = Config.quick(dataset="synthetic", controls=True, out_dir=str(out), fig_dir=str(out / "f"))
    return reproduce(cfg), out


def test_the_pipeline_produces_every_result_and_figure(smoke):
    results, out = smoke
    for key in ("logistic_3v7", "softmax", "maxout", "maxout_adv_trained", "rbf", "transfer"):
        assert key in results
    assert (
        json.loads((out / "results.json").read_text())["meta"]["config"]["dataset"] == "synthetic"
    )
    for fig in ("fig2_logistic", "fig3_maxout_filters", "fig4_epsilon_sweep", "error_vs_epsilon"):
        assert (out / "f" / f"{fig}.png").stat().st_size > 1000


def test_the_closed_form_logistic_attack_matches_autodiff(smoke):
    results, _ = smoke
    assert results["logistic_3v7"]["autodiff_matches_closed_form"] > 0.99


def test_every_paper_number_has_a_measured_counterpart_and_a_label():
    assert set(PAPER) == set(MEASURED) == set(LABELS)


def test_comparison_table_covers_every_paper_number(smoke):
    results, _ = smoke
    rows = comparison_rows(results)
    assert len(rows) == len(PAPER) and all(r[2] != "not run" for r in rows)
    assert markdown_table(results).split("\n\n")[0].count("\n") == len(PAPER) + 1
    assert "PGD" in markdown_table(results)
    assert all(not math.isinf(float(r[3])) for r in rows if r[3])


def test_cli(smoke, tmp_path, capsys):
    _, out = smoke
    assert main(["report", str(out / "results.json")]) == 0
    assert "Paper" in capsys.readouterr().out
    args = build_parser().parse_args(["reproduce", "--quick", "--dataset", "synthetic"])
    assert args.quick and args.dataset == "synthetic"
