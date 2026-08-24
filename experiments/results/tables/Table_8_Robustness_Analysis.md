# Table 8: Synthetic Robustness Analysis under Controlled Missingness & Noise

| Experiment_Type               | Perturbation_Level   |   Silhouette |   Davies_Bouldin |   Transition_Entropy_Bits |   Confidence_Pct |
|:------------------------------|:---------------------|-------------:|-----------------:|--------------------------:|-----------------:|
| Baseline (Clean)              | 0%                   |       0.0614 |           2.3944 |                     0     |            100   |
| Missingness Simulation (MCAR) | 10%                  |       0.1944 |           1.3936 |                     0     |            100   |
| Missingness Simulation (MCAR) | 20%                  |       0.282  |           0.9801 |                     0.042 |             98.8 |
| Missingness Simulation (MCAR) | 30%                  |       0.2788 |           1.0538 |                     0     |            100   |
| Gaussian Noise Simulation     | sigma=0.05           |       0.064  |           2.3357 |                     0     |            100   |
| Gaussian Noise Simulation     | sigma=0.1            |       0.0729 |           2.1763 |                     0     |            100   |
| Gaussian Noise Simulation     | sigma=0.2            |       0.0847 |           1.9522 |                     0     |            100   |