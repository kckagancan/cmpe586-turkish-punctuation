# Corpus-anchored supplemental transfer: verified results

Completed **2026-11-04**: **52/52 model-condition forwards and104 cohort score sets**.
All saved file hashes, original row IDs/order/targets, both-cohort metrics, own-anchor
drops, input exposure and shared batch work were independently reconciled locally.
No training, new checkpoint selection or repeated inference was used in analysis.

## What this adds

The existing selected models are evaluated on elongation, one internal omission and
partial folding, outside the explicit training-v1 augmentation operators. These are
new controlled distortions of already scored project test texts, **not a new natural
corpus test**. MultiLexNorm Turkish train-only morphology anchors the fixed recipe;
its test file remains unread. Sources, condition set and doses were frozen before
the new scores. The extension follows instructor feedback and the original test,
so it is a supplemental follow-up, not blind preregistration.

## Complete macro-F1 matrix

Values are punctuation macro-F1 (%) over PERIOD, COMMA and QUESTION_MARK.
Noisy cells are mean ± sample SD over seeds586/587/588; clean anchors have one value.
There is one training seed: these SDs measure corruption variation only.

### Primary:14,600 rows /181,485 targets

| Input | Published | Clean-trained | Augmented | Without case loss |
| --- | ---: | ---: | ---: | ---: |
| Clean anchor | 84.0071 | 84.3437 | 84.4824 | 84.3919 |
| Elongation | 83.7501 ± 0.0200 | 84.1030 ± 0.0302 | 84.3520 ± 0.0756 | 84.2705 ± 0.0785 |
| Internal omission | 83.8678 ± 0.0219 | 84.1801 ± 0.0153 | 84.4092 ± 0.0681 | 84.2990 ± 0.0319 |
| Partial folding | 83.8992 ± 0.0290 | 84.2450 ± 0.0144 | 84.4332 ± 0.0559 | 84.3753 ± 0.0171 |
| Matched full folding | 83.9514 ± 0.0278 | 84.2854 ± 0.0467 | 84.4134 ± 0.0144 | 84.3815 ± 0.0303 |

### Sensitivity:14,445 rows /181,116 targets

| Input | Published | Clean-trained | Augmented | Without case loss |
| --- | ---: | ---: | ---: | ---: |
| Clean anchor | 83.9760 | 84.2972 | 84.4389 | 84.3479 |
| Elongation | 83.7184 ± 0.0196 | 84.0554 ± 0.0306 | 84.3074 ± 0.0756 | 84.2252 ± 0.0782 |
| Internal omission | 83.8356 ± 0.0206 | 84.1325 ± 0.0138 | 84.3651 ± 0.0675 | 84.2547 ± 0.0320 |
| Partial folding | 83.8674 ± 0.0291 | 84.1981 ± 0.0144 | 84.3893 ± 0.0562 | 84.3315 ± 0.0173 |
| Matched full folding | 83.9200 ± 0.0282 | 84.2389 ± 0.0471 | 84.3694 ± 0.0143 | 84.3377 ± 0.0306 |

## Predefined paired contrasts

Differences are computed within each corruption seed before mean/SD aggregation (pp).
The primary contrast is augmented minus clean for each new mechanism; the matched
full-folding condition is a seen-mechanism control. All outcomes are retained.

| Input | Augmented − clean: primary | Augmented − clean: sensitivity | Augmented − without case loss: primary | Augmented − without case loss: sensitivity |
| --- | ---: | ---: | ---: | ---: |
| Clean anchor | 0.1387 | 0.1417 | 0.0905 | 0.0910 |
| Elongation | 0.2491 ± 0.0843 | 0.2521 ± 0.0845 | 0.0816 ± 0.0357 | 0.0823 ± 0.0359 |
| Internal omission | 0.2290 ± 0.0635 | 0.2326 ± 0.0640 | 0.1102 ± 0.0665 | 0.1104 ± 0.0670 |
| Partial folding | 0.1882 ± 0.0425 | 0.1913 ± 0.0427 | 0.0579 ± 0.0395 | 0.0578 ± 0.0397 |
| Matched full folding | 0.1279 ± 0.0446 | 0.1305 ± 0.0448 | 0.0318 ± 0.0345 | 0.0317 ± 0.0349 |

| Model | Partial − matched full: primary (pp) | Sensitivity (pp) |
| --- | ---: | ---: |
| Published | -0.0522 ± 0.0563 | -0.0526 ± 0.0567 |
| Clean-trained | -0.0404 ± 0.0361 | -0.0408 ± 0.0365 |
| Augmented | 0.0198 ± 0.0424 | 0.0200 ± 0.0428 |
| Without case loss | -0.0062 ± 0.0368 | -0.0062 ± 0.0371 |

The augmented model has a positive mean advantage over clean training for all
three new mechanisms, but the gains are small at these low anchors. Its clean-anchor
advantage is already0.1387pp: absolute noisy-score gains must be distinguished from
reduced clean-to-noisy degradation. The aggregate CSV retains own-anchor drops for
every metric/model/cohort. This is descriptive transfer under the specified recipe,
not evidence of significance, convergence or general natural-text robustness.

Partial-minus-full differences have mixed signs across models. Matching selected words
does **not** match character dose: partial changes2,171/2,188/2,140 characters versus
4,762/4,772/4,777 for full folding. These differences cannot isolate mechanism novelty.

## Class support and F1

Supports are invariant across models/conditions: primary O152,628; PERIOD15,465;
COMMA12,164; QUESTION_MARK1,228. Sensitivity O152,428; PERIOD15,325; COMMA12,152;
QUESTION_MARK1,211. Each JSON score set retains all four classes and its4×4 confusion.

| Input | Model | PERIOD F1 (%) | COMMA F1 (%) | QUESTION_MARK F1 (%) |
| --- | --- | ---: | ---: | ---: |
| Clean anchor | Published | 96.0549 | 63.2140 | 92.7524 |
| Clean anchor | Clean-trained | 96.1451 | 63.9560 | 92.9301 |
| Clean anchor | Augmented | 96.1348 | 63.9301 | 93.3824 |
| Clean anchor | Without case loss | 96.1359 | 63.7663 | 93.2735 |
| Elongation | Published | 96.0136 ± 0.0071 | 62.6029 ± 0.1826 | 92.6338 ± 0.1331 |
| Elongation | Clean-trained | 96.1093 ± 0.0099 | 63.3358 ± 0.1222 | 92.8638 ± 0.1172 |
| Elongation | Augmented | 96.0856 ± 0.0223 | 63.6261 ± 0.1286 | 93.3444 ± 0.2489 |
| Elongation | Without case loss | 96.1185 ± 0.0306 | 63.5136 ± 0.0717 | 93.1792 ± 0.1931 |
| Internal omission | Published | 96.0334 ± 0.0090 | 62.9216 ± 0.0445 | 92.6485 ± 0.0237 |
| Internal omission | Clean-trained | 96.1164 ± 0.0077 | 63.6009 ± 0.0861 | 92.8231 ± 0.0442 |
| Internal omission | Augmented | 96.1266 ± 0.0023 | 63.7550 ± 0.1001 | 93.3460 ± 0.1112 |
| Internal omission | Without case loss | 96.1231 ± 0.0081 | 63.6072 ± 0.0574 | 93.1667 ± 0.0442 |
| Partial folding | Published | 96.0442 ± 0.0101 | 63.0241 ± 0.0738 | 92.6293 ± 0.0288 |
| Partial folding | Clean-trained | 96.1257 ± 0.0032 | 63.7425 ± 0.0518 | 92.8669 ± 0.0171 |
| Partial folding | Augmented | 96.1174 ± 0.0035 | 63.8579 ± 0.1266 | 93.3242 ± 0.0869 |
| Partial folding | Without case loss | 96.1411 ± 0.0098 | 63.6678 ± 0.0230 | 93.3170 ± 0.0410 |
| Matched full folding | Published | 96.0401 ± 0.0097 | 63.0929 ± 0.0131 | 92.7213 ± 0.0617 |
| Matched full folding | Clean-trained | 96.1309 ± 0.0111 | 63.7788 ± 0.0665 | 92.9466 ± 0.0643 |
| Matched full folding | Augmented | 96.1196 ± 0.0003 | 63.8326 ± 0.0593 | 93.2879 ± 0.0269 |
| Matched full folding | Without case loss | 96.1387 ± 0.0112 | 63.7034 ± 0.0194 | 93.3025 ± 0.0853 |

## Dose, controls and audit scope

Eligible-opportunity rates are89/4,512 elongation,50/4,174 omission and43/617
operational partial folding. Exact primary edited words per seed are3,355,1,826 and
1,803, respectively; the full-folding control edits the same1,803 selected words.
These finite-corpus conditional anchors are transported to a different domain, not
population prevalence. Uniform whole-split selection simplifies natural tweet clustering.
The earlier case-neutral43/621 profile remains unchanged; four casing-incompatible
opportunities and zero hits were excluded for the operational partial-folding map.

All four fresh clean anchors exactly match their original held-out metrics in both
cohorts. Maximum encoding length154; no truncation or condition-specific row drop.
Sensitivity filters the same primary inputs/predictions without resampling. No unit
suite was added or run. Numerical checks used saved predictions and original targets;
the actual T4/FP32/SDPA-math runtime and selected tensors were verified before inference.

## Files and identities

- [All aggregate metrics, own-anchor drops and per-seed paired values](transfer-evaluation-aggregates.csv).
- [Every condition's P/R/F1, supports, confusions and exposure](transfer-evaluation-condition-summary.json).
- [Class P/R/F1/support in flat CSV](transfer-evaluation-per-class.csv).
- [Input coverage](transfer-evaluation-input-coverage.csv), [protocol](transfer-evaluation-protocol.md),
  [execution](transfer-evaluation-execution.md), [completion/archive](transfer-evaluation-completion.md).
- [Original study and version retention](report-versions.md); original metrics are unchanged.

Run: `8cf43b9d6a166baf005fb0320a52177f859c4b6215d06baaa0f93be57156d200`.
Analysis: `7847a6b85dcf7d4f1296cff6495974bcbf4103c635b628fce2d8820a45ef06f6`.

Dataset terms, exact split provenance/document independence and prior published-model
exposure remain unresolved. No p-values, equivalence claim, pooled robustness score or
new training-seed uncertainty estimate is introduced.
