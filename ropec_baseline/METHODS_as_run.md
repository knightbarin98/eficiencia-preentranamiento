# Methods (as-run) — for pasting into the paper

> Frozen, factual account of what was actually run. Numbers are from the real runs
> (5-seed contingency @100% and the 3-seed label-efficiency curve). No projected or
> expected values. Fill the software versions from any `versions.json` if you prefer
> different rounding; the values below are the recorded ones.

## Dataset and label unit

We used the PlaTiF tibial-plateau radiograph collection (186 patients; one MATLAB v5
`.mat` file per patient). Each file contains a variable number of radiographic views
(`im0…imN`, 1–8 per patient) plus a coronal CT field. We iterated all `imN` views and
used only the radiographic `OriginalImage` of each view; the coronal CT field was
excluded as out of protocol. Per-view Schatzker labels were decoded as fracture-present
for classes 1–6 and normal for class 7 (`y = 1[label ≠ 7]`), verified against the
clinical metadata spreadsheet (agreement on all 186 patients). The full corpus comprised
**421 views over 186 patients**.

Prediction and evaluation were performed at the **knee level**. Each knee unit was
defined programmatically as `knee_id = (patient_id, "fx" if label ≤ 6 else "normal")`,
yielding **190 knee units (128 fracture / 62 normal; knee-level prevalence 0.674)**.
Four patients contributed two knee units each (one fractured, one contralateral normal;
patient IDs 92, 112, 133, 147); the remaining 182 patients contributed a single unit. A
guard assertion enforced that the set of mixed-laterality patients matched exactly this
set before any modeling.

## Quality control

QC criteria were frozen before any metric was inspected: minimum resolution, image–mask
concordance, near-duplicate detection, and admitted-view type. Near-duplicate detection
(64-bit DCT perceptual hash, Hamming ≤ 5) was restricted to **cross-patient** matches,
since folds are patient-grouped and within-patient repeats (contralateral or multi-view)
carry no leakage risk and are legitimate data. QC removed **2 views** (patients 47 and
81) for image–mask shape mismatch; no cross-patient duplicates were found. The analysis
cohort after QC was **186 patients / 190 knee units / 419 views** (no knee units
dropped).

## Partitioning

We used 5-fold `StratifiedGroupKFold` over `patient_id`, stratified by patient-level
label, so that all views and both knee units of any patient fall in the same fold (zero
leakage, verified by an automated test). Within each training fold, an inner
patient-grouped split held out a validation set for early stopping and threshold
selection; the test fold was never used for model selection.

## Model, weights, and preprocessing

All comparators shared a single **ResNet-50** backbone (`timm`) with a single-logit head,
trained with `BCEWithLogitsLoss` using a positive-class weight equal to the train-fold
negative/positive view ratio. Provenance was the only difference between comparators:
**ImageNet** (timm pretrained), **RadImageNet** (loaded from the RadImageNet ResNet-50
checkpoint), and **random** initialization. Every weight load asserted the number of
successfully matched tensors; the RadImageNet checkpoint was remapped from its
`backbone.{0,1,4,5,6,7}` Sequential naming to torchvision `conv1/bn1/layer1..4` names and
required exactly **318 loaded tensors** or aborted (the reinitialized classifier head is
the only uninitialized module, as expected).

Each view's `OriginalImage` was intensity-normalized by its 1st–99th percentiles to
[0,1], bilinearly resized to 224×224, replicated to three channels, and standardized with
ImageNet statistics. This preprocessing was identical across all comparators. No data
augmentation was used: a controlled augmentation variant (flip, small affine, mild
brightness/contrast, plus head dropout) was tested and reduced out-of-fold performance
and widened the confidence intervals on this small cohort, so it was not adopted.

## Training

Identical for all comparators: Adam (learning rate 3×10⁻⁴, weight decay 1×10⁻⁴), batch
size 32, up to 30 epochs with early stopping (patience 7) on the inner-validation AUROC,
restoring the best checkpoint. Optimizer, schedule, epoch budget, and tuning were shared
across comparators; the per-fold random seed was identical across comparators so that
folds, batch order, and head initialization were matched.

## View-to-knee aggregation and metrics

Out-of-fold view logits were aggregated to the knee level by the **mean of logits within
each `knee_id`** (frozen before any metric was computed); the two knee units of a mixed
patient were never averaged together. The primary metric was **out-of-fold AUROC at the
knee level**. Provenance contrasts were quantified as **ΔAUROC with a paired cluster
bootstrap that resamples whole patients** (2000 resamples; each patient carries its 1–2
knee units together), reporting the observed effect, 95% percentile interval, and a
two-sided bootstrap p-value. We also report AUPRC and prevalence. Brier score /
calibration were not claimed.

## Reproducibility and multi-seed protocol

For each run we versioned the fold manifest, QC exclusion log, configuration with seed,
per-fold out-of-fold knee predictions, and the software environment. Robustness was
assessed over independent seeds: the @100% contingency used 5 seeds
(1337, 2024, 7, 42, 2718) and the label-efficiency curve used 3 seeds (1337, 2024, 7).
Absolute AUROC is reported as mean ± SD across seeds (optimization variability); ΔAUROC
intervals pool the per-seed bootstrap distributions (sampling and seed variability
combined).

## Label-efficiency curve

Nested, class-stratified subsets of the **training** patients (10% ⊂ 25% ⊂ 50% ⊂ 100%)
were drawn per fold and seed and were identical across comparators; validation and test
sets were unchanged across fractions. This tests the provenance × fraction interaction.

## Software

Python 3.11.15; PyTorch 2.13.0 (CUDA 13.0, cuDNN 9.2.0); timm 1.0.28; scikit-learn 1.9.0;
NumPy 2.4.6; pandas 3.0.3; SciPy 1.17.1; openpyxl 3.1.5. Hardware: NVIDIA GeForce RTX
5060 Ti (16 GB).

---

# Results (as-run numbers — factual)

## Provenance at 100% labels (5 seeds; OOF knee-level AUROC, mean ± SD)

| Comparator | AUROC | AUPRC |
|---|---|---|
| RadImageNet | 0.679 ± 0.038 | 0.800 ± 0.024 |
| ImageNet | 0.664 ± 0.028 | 0.784 ± 0.021 |
| random | 0.526 ± 0.020 | 0.698 ± 0.022 |

ΔAUROC (paired patient bootstrap; mean ± SD across seeds · 95% pooled CI · bootstrap p):

- RadImageNet − ImageNet: +0.015 ± 0.036 · [−0.109, +0.136] · p = 0.80 (null)
- RadImageNet − random: +0.154 ± 0.057 · [+0.009, +0.286] · p = 0.035
- ImageNet − random: +0.139 ± 0.039 · [+0.011, +0.267] · p = 0.035

## Label-efficiency curve (3 seeds; OOF knee-level AUROC, mean ± SD)

| Fraction (train patients) | RadImageNet | ImageNet | random |
|---|---|---|---|
| 10% (13) | 0.506 ± 0.022 | 0.550 ± 0.045 | 0.511 ± 0.024 |
| 25% (31) | 0.545 ± 0.030 | 0.547 ± 0.029 | 0.535 ± 0.038 |
| 50% (61) | 0.621 ± 0.027 | 0.634 ± 0.011 | 0.544 ± 0.035 |
| 100% (122) | 0.688 ± 0.006 | 0.663 ± 0.031 | 0.526 ± 0.007 |

Provenance × fraction interaction (RadImageNet − ImageNet ΔAUROC): −0.044 (10%),
−0.003 (25%), −0.013 (50%), +0.025 (100%); the 95% interval crosses zero at every
fraction (p between 0.53 and 0.99). The pretraining-over-random advantage reached
significance only at 100% labels (RadImageNet − random and ImageNet − random, both
p < 0.05), and was null at 10–50%.

## Statement of findings (factual)

At full label budget, domain-specific pretraining (RadImageNet) showed no detectable
advantage over natural-image pretraining (ImageNet) at the knee level, while both
outperformed random initialization. Across the label-efficiency curve the
RadImageNet−ImageNet gap did not widen at lower label fractions (null at all budgets),
and the pretraining-over-random benefit was concentrated at the full budget rather than
at scarce labels. At 10–25% of training patients (13–31 patients) all initializations
performed near chance.
