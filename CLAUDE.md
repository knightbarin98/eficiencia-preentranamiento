# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A controlled ablation study (thesis / ROPEC 2026 paper) measuring the effect of
**pretraining provenance** on tibial-plateau fracture detection. A single
**ResNet-50** backbone is compared under three weight inits — **ImageNet vs
RadImageNet vs random** — on the **PlaTiF** dataset, evaluated at the **knee level**
(fracture present/absent). Primary metric is **out-of-fold (OOF) AUROC**; the
provenance contrast is reported as **ΔAUROC with a paired cluster bootstrap that
resamples whole patients** (effect + 95% CI + bootstrap p), never binary significance.

This repo has two sides: the **experiment code** (`ropec_baseline/`, below) and the
**manuscript** (planning docs + LaTeX kit, see "Manuscript" below). The prose/planning
`.md` files at the repo root (`PAPER_ROPEC_*`, `ROPEC2026_*`) are the paper draft and
research plan, in Spanish. Code comments are also in Spanish — match that when editing.
The paper's claim is a **null result**: RadImageNet ≈ ImageNet (provenance does not
govern label efficiency); both beat random only at sufficient labels. Never write it up
as a superiority result, and never invent/project numbers — see the "NO afirmar" list in
`PAPER_ROPEC_encuadre_y_outline.md`.

## Where the code is

All working code lives in **`ropec_baseline/`**. The repo-root `src/`, `tests/`,
`docs/`, `config/`, `Dockerfile`, `Makefile`, `setup.py`, and `requirements.txt` are
empty scaffolding — ignore them. `ropec_baseline/` has its own `requirements.txt`,
`config.yaml`, and `tests/`.

## Environment & commands

No existing conda env has the full stack — create a fresh one:

```bash
conda create -n ropec python=3.11 -y && conda activate ropec
cd ropec_baseline && pip install -r requirements.txt
```

Run everything from inside `ropec_baseline/`:

```bash
# Single CV run (all 3 comparators), driven entirely by config.yaml
python run_experiment.py --config config.yaml
python run_experiment.py --config config.yaml --smoke   # 1 outer fold, fast plumbing check

# Multi-seed robustness sweep (the @100% contingency result)
python run_multiseed.py --config config.yaml            # --smoke for 1 fold/seed

# Label-efficiency curve (provenance × fraction)
python run_curve.py --config config.yaml                # --smoke for 1 seed/1 fold

# Standalone QC of the real cohort
python qc.py --config config.yaml [--out-dir DIR]

# Figures / tables (Phase 5)
python make_figures.py --curve outputs/curve/curve_summary.json \
                       --multiseed outputs/multiseed_100/multiseed_summary.json

# Zero-leakage tests (run on synthetic data, no real data needed)
pytest tests/                    # or: python tests/test_splits.py  (runs standalone)
```

`toy` vs `real` mode is set by `mode:` in `config.yaml`. Toy mode synthesizes a
PlaTiF-shaped dataset (variable views/patient, ≥1 mixed knee) so the whole pipeline —
including `knee_id` derivation and assertions — runs with no real data on disk.

## Architecture & data flow

`run_experiment.run_once()` is the core single-seed CV run and is **reused** by
`run_multiseed.py` and `run_curve.py` (they loop it over seeds/fractions). The flow:

`data.build_index` → `splits.make_folds` (+ `check_no_leakage`) → `train.train_one_fold`
per comparator → `eval.aggregate_views_to_knee` → `eval.knee_metrics` +
`eval.paired_bootstrap_delta_auroc` → artifacts under `outputs/<run_name>/`.

Module roles: `config.yaml` (all parameterization; absolute data paths, outputs in
repo), `utils.py` (seeds, logging, version/config dump), `data.py` (index + Dataset;
toy and real PlaTiF `.mat` v5 loading), `splits.py` (patient-grouped folds + leakage
guards + manifest), `model.py` (ResNet-50 + asserted weight loading), `train.py`
(per-fold loop, early stopping, threshold), `eval.py` (view→knee aggregation, metrics,
bootstrap), `qc.py` / `failure_analysis.py` / `make_figures.py` (analysis).

Each run writes `index.csv`, `folds_manifest.json`, per-comparator
`<comparator>/{fold_k_preds.csv, oof_knee_preds.csv, metrics.json}`, plus
`delta_auroc.json`, `summary.json`, `versions.json`, `config_used.yaml`, `run.log`.

## Invariants — do not violate these

These are the scientific integrity guardrails of the study; they are enforced by
assertions and must stay enforced:

1. **Knee-level unit.** `knee_id = (patient_id, "fx" if label ≤ 6 else "normal")`
   (Schatzker 1–6 = fracture, 7 = normal). View→knee aggregation is the **mean of
   logits** within a `knee_id`, frozen before any metric is computed. The two knee
   units of a mixed-laterality patient are never averaged together.
2. **Zero leakage.** `StratifiedGroupKFold(5)` over `patient_id` — all views and both
   knees of a patient in the same fold. An inner patient-grouped split handles early
   stopping and threshold; the test fold is never used for selection. Enforced by
   `splits.check_no_leakage` and `tests/test_splits.py`.
3. **Single backbone, matched everything.** ResNet-50 is the only allowed backbone
   (asserted). All comparators share folds, augs, optimizer, epochs, tuning budget,
   and per-fold seed — provenance is the *only* difference.
4. **Asserted weight loading.** Every checkpoint load asserts the matched-tensor count.
   RadImageNet ResNet-50 is remapped from its `backbone.{0,1,4,5,6,7}` naming to
   torchvision names and requires **exactly 318 tensors or aborts**.
5. **Real-cohort ground truth is asserted.** `data.assert_real_index` locks the cohort
   (186 patients / 190 knees / 419 views post-QC, mixed-side patient IDs 92/112/133/147).
   QC criteria are frozen before metrics are inspected (`dup_scope: cross_patient` —
   only cross-patient perceptual-hash duplicates count as leakage).
6. **No invented numbers.** Results and the Methods "as-run" text are written only from
   real run outputs. `METHODS_as_run.md` is a frozen factual account — treat recorded
   numbers there as authoritative, don't project or fabricate.

## Manuscript (paper deliverable)

The paper is an **IEEE conference 6-pager for ROPEC 2026**. Two layers:

- **Planning / content (Spanish), repo root** — write from these, don't re-derive:
  - `PAPER_ROPEC_encuadre_y_outline.md` — the authoritative framing: exact real numbers
    to cite verbatim, semi-assembled abstract, intro/related-work skeleton with reading
    slots, 6-page IEEE structure, and the "NO afirmar" integrity guardrails.
  - `ROPEC2026_reposicionamiento_v2.md`, `ROPEC2026_ModuloA_plan_y_lecturas_v2.md`,
    `ROPEC2026_posicionamiento_y_plan_diario_v2.md` — positioning vs. the direct neighbor
    (Elnakib et al. 2026, arXiv:2606.17295 — *unsupervised* SSL phenotyping on PlaTiF; a
    different task from this *supervised* provenance ablation) and the daily plan.
  - `ropec_baseline/METHODS_as_run.md` — frozen factual Methods+Results to paste into the
    paper; the recorded numbers there are authoritative.

- **LaTeX template, `ROPEC2026LatexTemplateKit/`** — `ROPEC2026LatexTemplate.tex` is the
  IEEEtran `conference` skeleton (currently unmodified `bare_conf` boilerplate — the real
  manuscript still has to be written into it). Bundled: `IEEEtran.cls`, `IEEEtranBST2/`
  (`.bst` + example `.bib`), `fig1.png`. Real figures live in
  `ropec_baseline/outputs/figures/` (`fig_efficiency_curve.pdf`, `fig_forest_delta100.pdf`).

Build the PDF (from inside the kit dir; `.aux/.log/.pdf/.synctex.gz` are build artifacts):

```bash
cd ROPEC2026LatexTemplateKit
pdflatex ROPEC2026LatexTemplate.tex && bibtex ROPEC2026LatexTemplate && \
  pdflatex ROPEC2026LatexTemplate.tex && pdflatex ROPEC2026LatexTemplate.tex
```

## Phase roadmap

Phase 0 = scaffolding + toy smoke. Phase 1 = wire real PlaTiF (rewrite
`build_real_index()` in `data.py`). Phase 2 = RadImageNet contingency @100% →
checkpoint/minimal paper (freeze + `git tag`). Phase 3 = label-efficiency curve.
Phase 5 = figures, tables, as-run Methods. See `ropec_baseline/README.md` for detail.
