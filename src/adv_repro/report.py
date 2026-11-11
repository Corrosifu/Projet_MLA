"""Side-by-side table: what the paper reports versus what this repository measures."""

from __future__ import annotations

import math

from adv_repro.paper import PAPER

# paper key -> function(results) -> value measured here
MEASURED = {
    "logistic_3v7_clean_error": lambda r: r["logistic_3v7"]["clean_error"],
    "logistic_3v7_adv_error": lambda r: r["logistic_3v7"]["adv_error"],
    "softmax_adv_error": lambda r: r["softmax"]["adv_error"],
    "softmax_adv_confidence": lambda r: r["softmax"]["adv_confidence_on_mistakes"],
    "maxout_clean_error": lambda r: r["maxout"]["clean_error"],
    "maxout_adv_error": lambda r: r["maxout"]["adv_error"],
    "maxout_adv_confidence": lambda r: r["maxout"]["adv_confidence_on_mistakes"],
    "advtrained_clean_error": lambda r: r["maxout_adv_trained"]["clean_error"],
    "advtrained_adv_error": lambda r: r["maxout_adv_trained"]["adv_error"],
    "advtrained_adv_confidence": lambda r: r["maxout_adv_trained"]["adv_confidence_on_mistakes"],
    "transfer_advtrained_to_original_error": lambda r: r["maxout_adv_trained"][
        "transfer_to_standard_error"
    ],
    "transfer_original_to_advtrained_error": lambda r: r["maxout"]["transfer_to_adv_trained_error"],
    "noise_sign_adv_error": lambda r: r["noise_sign"]["adv_error"],
    "noise_sign_adv_confidence": lambda r: r["noise_sign"]["adv_confidence_on_mistakes"],
    "noise_uniform_adv_error": lambda r: r["noise_uniform"]["adv_error"],
    "noise_uniform_adv_confidence": lambda r: r["noise_uniform"]["adv_confidence_on_mistakes"],
    "rbf_adv_error": lambda r: r["rbf"]["adv_error"],
    "rbf_adv_confidence": lambda r: r["rbf"]["adv_confidence_on_mistakes"],
    "rbf_clean_confidence": lambda r: r["rbf"]["clean_confidence"],
    "rubbish_softmax_error": lambda r: r["rubbish"]["softmax"]["error_rate"],
    "rubbish_softmax_confidence": lambda r: r["rubbish"]["softmax"]["confidence_on_mistakes"],
    "rubbish_maxout_error": lambda r: r["rubbish"]["maxout"]["error_rate"],
    "rubbish_maxout_confidence": lambda r: r["rubbish"]["maxout"]["confidence_on_mistakes"],
    "rubbish_rbf_error": lambda r: r["rubbish"]["rbf"]["error_rate"],
    "transfer_maxout_to_softmax_agreement": lambda r: r["transfer"]["maxout_to_softmax_agreement"],
    "transfer_maxout_to_rbf_agreement": lambda r: r["transfer"]["maxout_to_rbf_agreement"],
    "transfer_maxout_to_softmax_agreement_both_wrong": lambda r: r["transfer"][
        "maxout_to_softmax_agreement_both_wrong"
    ],
    "transfer_maxout_to_rbf_agreement_both_wrong": lambda r: r["transfer"][
        "maxout_to_rbf_agreement_both_wrong"
    ],
}

LABELS = {
    "logistic_3v7_clean_error": "Logistic regression 3 vs 7: clean error",
    "logistic_3v7_adv_error": "Logistic regression 3 vs 7: error on FGSM examples",
    "softmax_adv_error": "Softmax regression: error on FGSM examples",
    "softmax_adv_confidence": "Softmax regression: confidence on those mistakes",
    "maxout_clean_error": "Maxout + dropout: clean test error",
    "maxout_adv_error": "Maxout: error on FGSM examples",
    "maxout_adv_confidence": "Maxout: confidence on those mistakes",
    "advtrained_clean_error": "Adversarially trained maxout: clean test error",
    "advtrained_adv_error": "Adversarially trained maxout: error on FGSM examples",
    "advtrained_adv_confidence": "Adversarially trained maxout: confidence on mistakes",
    "transfer_original_to_advtrained_error": "FGSM examples of original maxout: error on adv. trained",
    "transfer_advtrained_to_original_error": "FGSM examples of adv. trained maxout: error on original",
    "noise_sign_adv_error": "Control, random +-eps noise: error on FGSM examples",
    "noise_sign_adv_confidence": "Control, random +-eps noise: confidence on mistakes",
    "noise_uniform_adv_error": "Control, uniform noise: error on FGSM examples",
    "noise_uniform_adv_confidence": "Control, uniform noise: confidence on mistakes",
    "rbf_adv_error": "Shallow RBF network: error on FGSM examples",
    "rbf_adv_confidence": "Shallow RBF network: confidence on those mistakes",
    "rbf_clean_confidence": "Shallow RBF network: average confidence on clean data",
    "rubbish_softmax_error": "Gaussian noise inputs, softmax regression: error",
    "rubbish_softmax_confidence": "Gaussian noise inputs, softmax regression: confidence",
    "rubbish_maxout_error": "Gaussian noise inputs, maxout: error",
    "rubbish_maxout_confidence": "Gaussian noise inputs, maxout: confidence",
    "rubbish_rbf_error": "Gaussian noise inputs, RBF: error",
    "transfer_maxout_to_softmax_agreement": "Maxout's class predicted by softmax regression",
    "transfer_maxout_to_rbf_agreement": "Maxout's class predicted by the RBF network",
    "transfer_maxout_to_softmax_agreement_both_wrong": "... same, among cases where both are wrong (softmax)",
    "transfer_maxout_to_rbf_agreement_both_wrong": "... same, among cases where both are wrong (RBF)",
}


def _pct(v: float | None) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.1f} %"


def comparison_rows(results: dict) -> list[tuple[str, str, str, str, str]]:
    """(metric, paper, measured here, difference in points, where in the paper)."""
    rows = []
    for key, (paper_value, where) in PAPER.items():
        try:
            mine = MEASURED[key](results)
        except KeyError:  # e.g. the noise controls were not run
            rows.append((LABELS[key], _pct(paper_value), "not run", "", where))
            continue
        diff = "" if mine is None or math.isnan(mine) else f"{100 * (mine - paper_value):+.1f}"
        rows.append((LABELS[key], _pct(paper_value), _pct(mine), diff, where))
    return rows


def pgd_table(results: dict) -> str:
    """Extra table (not from the paper): single-step versus iterative attack."""
    pgd = results.get("pgd")
    if not pgd:
        return ""
    lines = [
        f"\n\n**Beyond the paper: PGD ({pgd['steps']} steps) versus FGSM, eps = 0.25**\n",
        "| Model | FGSM error | PGD error |",
        "|---|---|---|",
    ]
    for name in ("maxout", "maxout_adv_trained"):
        lines.append(
            f"| {name} | {_pct(pgd[name]['fgsm_error'])} | {_pct(pgd[name]['pgd_error'])} |"
        )
    return "\n".join(lines)


def markdown_table(results: dict) -> str:
    lines = [
        "| Metric | Paper | This repo | Diff (pts) | Source |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {m} | {p} | {a} | {d} | {w} |" for m, p, a, d, w in comparison_rows(results)]
    return "\n".join(lines) + pgd_table(results)
