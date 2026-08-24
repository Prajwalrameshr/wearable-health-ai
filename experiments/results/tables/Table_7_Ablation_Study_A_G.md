# Table 7: Component Ablation Study (Steps A to G)

| Step                        | Description                                     |   Silhouette | Improvement_vs_Prev          |
|:----------------------------|:------------------------------------------------|-------------:|:-----------------------------|
| A: Raw Physiology           | Raw physiological signals only (GMM)            |       0.1724 | Baseline                     |
| B: + Baseline Deviations    | Added personal baseline deviations              |       0.1509 | -0.0214                      |
| C: + Temporal Slopes        | Added 7-day rolling slopes                      |       0.1118 | -0.0391                      |
| D: + Severity Score         | Added composite severity normalization          |       0.1944 | +0.0826                      |
| E: + GMM Soft Probabilities | Continuous GMM posterior probability vectors    |       0.1944 | 0.0000 (Soft Representation) |
| F: + Hard HMM               | Discrete state sequence HMM decoding            |       0.1944 | 0.0000 (Discrete HMM)        |
| G: + Soft HMM (Proposed)    | Continuous soft probability vector HMM decoding |       0.1944 | 0.0000 (Continuous Soft HMM) |