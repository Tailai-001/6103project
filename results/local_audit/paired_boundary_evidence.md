# Paired boundary sensitivity results

Negative differences favour the candidate. 95% intervals resample the original paired batches.
These intervals describe augmentation behaviour, not seed-level classification uncertainty.
Shape-close is a fixed observational subset, not an exactly shape-matched causal control.

| Comparison | Boundary pairs | Subset | N paired | Candidate − reference | 95% batch bootstrap CI |
|---|---|---|---:|---:|---|
| E3 − E2 | all | all_valid | 2550 | -0.05963 | [-0.06302, -0.05633] |
| E3 − E2 | all | no_new_image_edge | 2550 | -0.05963 | [-0.06302, -0.05633] |
| E3 − E2 | without_any_padding | all_valid | 2542 | -0.03675 | [-0.03925, -0.03418] |
| E3 − E2 | without_any_padding | no_new_image_edge | 2542 | -0.03675 | [-0.03925, -0.03418] |
| E4 − E3 | all | all_valid | 2550 | -0.03718 | [-0.03823, -0.03602] |
| E4 − E3 | all | no_new_image_edge | 2183 | -0.03702 | [-0.03819, -0.03582] |
| E4 − E3 | without_any_padding | all_valid | 2541 | -0.03949 | [-0.04091, -0.03802] |
| E4 − E3 | without_any_padding | no_new_image_edge | 2176 | -0.04100 | [-0.04235, -0.03961] |
| E4 − E4-R | all | all_valid | 2550 | -0.05538 | [-0.05710, -0.05369] |
| E4 − E4-R | all | no_new_image_edge | 2062 | -0.05659 | [-0.05792, -0.05526] |
| E4 − E4-R | all | accepted_shape_close | 197 | -0.05068 | [-0.05619, -0.04484] |
| E4 − E4-R | without_any_padding | all_valid | 2541 | -0.04657 | [-0.04797, -0.04518] |
| E4 − E4-R | without_any_padding | no_new_image_edge | 2055 | -0.04830 | [-0.04969, -0.04698] |
| E4 − E4-R | without_any_padding | accepted_shape_close | 197 | -0.04382 | [-0.04761, -0.03953] |

Full RGB discontinuity increased for 149/2342 accepted changed E4 masks.
The local DP objective therefore does not guarantee improvement of the complete final boundary metric.
The source is a 20-batch training audit with p=1; do not extrapolate these intervals to p=0.5 training accuracy.
