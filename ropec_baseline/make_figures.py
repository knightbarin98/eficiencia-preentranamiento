"""Fase 5 (Prompt 5.1): figuras y tabla publicables desde los JSON de resumen.

Lee:
  outputs/curve/curve_summary.json          (curva de eficiencia)
  outputs/<multiseed_run>/multiseed_summary.json  (@100%, 5 semillas)
Produce en outputs/figures/:
  fig_efficiency_curve.(png|pdf)   AUROC vs fracción, 3 comparadores ± SD
  fig_forest_delta100.(png|pdf)    ΔAUROC @100% con IC95% agrupado
  table_provenance_metrics.csv     procedencia × métricas (@100%)

Uso:
    python make_figures.py --config config.yaml
    python make_figures.py --curve outputs/curve/curve_summary.json \
                           --multiseed outputs/multiseed_100/multiseed_summary.json
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Paleta fija (consistente con el resto del análisis)
COLORS = {"radimagenet": "#2a78d6", "imagenet": "#008300", "random": "#898781"}
MARKERS = {"radimagenet": "o", "imagenet": "s", "random": "^"}
LABELS = {"radimagenet": "RadImageNet", "imagenet": "ImageNet", "random": "random"}


def _save(fig, path_noext):
    fig.savefig(path_noext + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(path_noext + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ->", path_noext + ".png / .pdf")


def figure_efficiency_curve(curve, out_dir):
    fractions = [float(f) for f in curve["fractions"]]
    xs = [f * 100 for f in fractions]
    comparators = curve["comparators"]
    ct = curve["curve"]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for c in comparators:
        means = [ct[str(f)][c]["auroc_mean"] for f in fractions]
        stds = [ct[str(f)][c]["auroc_std"] for f in fractions]
        ax.errorbar(xs, means, yerr=stds, marker=MARKERS.get(c, "o"), capsize=3,
                    lw=2, color=COLORS.get(c), label=LABELS.get(c, c))
    ax.axhline(0.5, ls="--", lw=1, color="#898781", alpha=0.7, label="chance (0.5)")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("training-label fraction (%)")
    ax.set_ylabel("OOF AUROC (knee level)")
    ax.set_ylim(0.45, 0.75)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Label efficiency by pretraining provenance", fontsize=11)
    _save(fig, os.path.join(out_dir, "fig_efficiency_curve"))


def figure_forest_delta100(multiseed, out_dir):
    pp = multiseed["per_pair"]
    order = [k for k in ["radimagenet_vs_imagenet", "radimagenet_vs_random",
                         "imagenet_vs_random"] if k in pp]
    names = {"radimagenet_vs_imagenet": "RadImageNet − ImageNet",
             "radimagenet_vs_random": "RadImageNet − random",
             "imagenet_vs_random": "ImageNet − random"}

    fig, ax = plt.subplots(figsize=(6, 2.8))
    for i, k in enumerate(order):
        d = pp[k]
        y = len(order) - 1 - i
        lo, hi = d["pooled_ci_low"], d["pooled_ci_high"]
        eff = d["delta_mean_over_seeds"]
        sig = d["pooled_ci_low"] > 0 or d["pooled_ci_high"] < 0
        color = "#185FA5" if sig else "#898781"
        ax.plot([lo, hi], [y, y], lw=2, color=color)
        ax.plot([eff], [y], "o", color=color)
        ax.text(hi + 0.01, y, f"Δ={eff:+.3f}  p={d['pooled_p_bootstrap']:.3f}",
                va="center", fontsize=8)
    ax.axvline(0, ls="--", lw=1, color="#444")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[k] for k in reversed(order)], fontsize=9)
    ax.set_xlabel("ΔAUROC (95% pooled CI, patient bootstrap)")
    ax.set_xlim(-0.2, 0.45)
    ax.set_title("Provenance contrasts @100% labels", fontsize=11)
    ax.grid(True, axis="x", alpha=0.25)
    _save(fig, os.path.join(out_dir, "fig_forest_delta100"))


def table_provenance_metrics(multiseed, out_dir):
    pc = multiseed["per_comparator"]
    rows = []
    for c in multiseed["comparators"]:
        rows.append({
            "comparator": LABELS.get(c, c),
            "auroc_mean": round(pc[c]["auroc_mean"], 4),
            "auroc_sd": round(pc[c]["auroc_std"], 4),
            "auprc_mean": round(pc[c]["auprc_mean"], 4),
            "auprc_sd": round(pc[c]["auprc_std"], 4),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "table_provenance_metrics.csv")
    df.to_csv(path, index=False)
    print("  ->", path)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--curve", default=None)
    ap.add_argument("--multiseed", default=None)
    args = ap.parse_args()

    out_root = "./outputs"
    if os.path.exists(args.config):
        import yaml
        out_root = yaml.safe_load(open(args.config))["paths"]["output_root"]

    curve_path = args.curve or os.path.join(out_root, "curve", "curve_summary.json")
    ms_path = args.multiseed or os.path.join(out_root, "multiseed_100", "multiseed_summary.json")
    out_dir = os.path.join(out_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(ms_path):
        multiseed = json.load(open(ms_path))
        print("== @100% multi-semilla ==")
        figure_forest_delta100(multiseed, out_dir)
        table_provenance_metrics(multiseed, out_dir)
    else:
        print(f"AVISO: no encontré {ms_path}")

    if os.path.exists(curve_path):
        curve = json.load(open(curve_path))
        print("== curva de eficiencia ==")
        figure_efficiency_curve(curve, out_dir)
    else:
        print(f"AVISO: no encontré {curve_path}")

    print(f"\nFiguras/tabla en: {out_dir}")


if __name__ == "__main__":
    main()
