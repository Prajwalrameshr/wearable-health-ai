# Table 5: Continuous Soft-Posterior HMM vs Discrete Categorical HMM

| Model Variant                    | Observation Input             |   Transition Entropy (bits) | Sequence Log-Likelihood   |   Mean State Duration (days) |   Transition Stability | Uncertainty Representation               |
|:---------------------------------|:------------------------------|----------------------------:|:--------------------------|-----------------------------:|-----------------------:|:-----------------------------------------|
| Hard-State Categorical HMM (B5)  | Discrete cluster integer      |                           0 | N/A (Discrete)            |                          4.2 |                 0.7619 | Discarded via hard argmax                |
| Proposed Soft-Posterior HMM (B6) | Continuous GMM posterior P(S) |                           0 | 19.9468                   |                         29   |                 1      | Preserved continuous probability simplex |