| Method | Type | Single (batch=1) ms/sample | Batched (N=100) ms/sample |
|---|---|---:|---:|
| HiT-IDS prediction only (no explanation) | reference | 1.820 | 0.019 |
| HiT-IDS attention rollout | intrinsic | 3.406 | 0.035 |
| HiT-IDS input x gradient | post-hoc (1 backward) | 7.516 | 0.081 |
| HiT-IDS GradientSHAP | post-hoc | 517.795 | 515.322 |
| Random Forest TreeSHAP | post-hoc | 295.118 | 294.683 |
| XGBoost TreeSHAP | post-hoc | 8.484 | 0.631 |
| BiLSTM GradientSHAP | post-hoc | 1066.274 | 1063.456 |
| Vanilla Transformer GradientSHAP | post-hoc | 348.011 | 346.214 |

Speed-up of attention rollout (same regime): HiT-IDS input x gradient: 2.2× single / 2.3× batched; HiT-IDS GradientSHAP: 152.0× single / 14573.7× batched; Random Forest TreeSHAP: 86.7× single / 8333.9× batched; XGBoost TreeSHAP: 2.5× single / 17.8× batched; BiLSTM GradientSHAP: 313.1× single / 30075.3× batched; Vanilla Transformer GradientSHAP: 102.2× single / 9791.2× batched
