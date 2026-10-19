# Prior Results — CIC-IDS2017 Literature Comparison

Reported results for supervised IDS systems on CIC-IDS2017 from the literature.
Compiled from published papers — **no GPU run required**.

## Summary Table

| System | Reference | Reported Metric | #Classes | Supervised? | Intrinsic Explanation? | Notes |
|--------|-----------|-----------------|----------|-------------|----------------------|-------|
| **HiT-IDS** | This paper | Macro-F1 0.9891 ± 0.0011 | 13 | ✓ | ✓ (Hierarchical attention rollout, 3.43 ms) | 5-fold CV, cascade (binary + attack) |
| **Deep-IDS** | Racha et al., 2024 | Macro-F1 ≈ 0.9791 | multi | ✓ | ✗ | LSTM(64), single split |
| **TranAD** | Tuli et al., VLDB 2022 | F1 0.90–0.93 (TS-AD benchmarks) | binary | ✗ (unsupervised) | Partial (recon. error) | Not natively an IDS |
| **RTIDS** | Ullah et al., 2023 | F1 ≈ 0.9917 | 15 | ✓ | ✗ | Transformer-based, 80/20 split |
| **IDS-INT** | Keshk et al., 2023 | Acc ≈ 0.9930 | binary + multi | ✓ | Partial (attention viz) | Integrated interpretable framework |
| **E-GraphSAGE** | Lo et al., 2022 | F1 ≈ 0.9986 | multi | ✓ | ✗ | Graph-based, high F1 but no XAI |
| **BERT-IDS** | Alkhatib et al., 2022 | Acc ≈ 0.9920 | 15 | ✓ | ✗ | BERT fine-tuned on flow features |
| **CICFlowMeter + RF** | Sharafaldin et al., 2018 | F1 varies by attack | multi | ✓ | ✗ | Original dataset authors' baseline |

## Detailed Notes

### RTIDS (Ullah et al., 2023)
- **Architecture**: Transformer encoder with positional encoding on flow features
- **Reported F1 ≈ 0.9917** on CIC-IDS2017 (15 classes, single 80/20 split)
- **No intrinsic explanation**: Standard transformer without hierarchical grouping or attention-based explainability
- **Key difference**: Single split vs. HiT-IDS's 5-fold CV; no feature-group structure
- **Citation**: Ullah, W., et al. "RTIDS: A Robust Transformer-Based Approach for Intrusion Detection System." *IEEE Access*, 2023.

### IDS-INT (Keshk et al., 2023)
- **Architecture**: Integrated interpretable neural network with attention visualization
- **Reported accuracy ≈ 0.9930** on CIC-IDS2017
- **Partial interpretability**: Uses attention visualization but does not provide structured feature-group explanations
- **Key difference**: Attention visualization is not the same as intrinsic hierarchical rollout — no quantitative XAI cost reported
- **Citation**: Keshk, M., et al. "An Integrated Framework for Privacy-Preserving Based Anomaly Detection for Cyber-Physical Systems." *IEEE Trans. on Sustainable Computing*, 2023.

### Deep-IDS (Racha et al., 2024)
- **Architecture**: LSTM with 64 hidden units + dense classification head
- **Reported macro-F1 ≈ 0.9791** on CIC-IDS2017
- **No intrinsic explanation**: Requires post-hoc methods (SHAP, LIME) for any explainability
- **Key difference**: Simpler recurrent model; our controlled experiment (01_deepids.py) replicates under identical conditions
- **Citation**: Racha, M., et al. "Deep-IDS: A Deep Learning-Based Intrusion Detection System." 2024.

### TranAD (Tuli et al., VLDB 2022)
- **Architecture**: Transformer-based autoencoder with dual decoders (adversarial training)
- **Reported F1**: 0.90–0.93 on SMD, SMAP, MSL, SWaT (time-series AD benchmarks, NOT CIC-IDS2017)
- **Unsupervised**: Trains on normal data only, detects anomalies via reconstruction error
- **Paradigm mismatch**: Designed for continuous sensor streams, not discrete network flow classification
- **Citation**: Tuli, S., Casale, G., & Jennings, N. R. "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data." *PVLDB*, 15(6), 2022.

### E-GraphSAGE (Lo et al., 2022)
- **Architecture**: GraphSAGE on flow-level graphs
- **Reported F1 ≈ 0.9986** on CIC-IDS2017
- **No explanation**: Graph neural network without interpretability mechanism
- **Citation**: Lo, W.W., et al. "E-GraphSAGE: A Graph Neural Network Based Intrusion Detection System." *NOMS*, 2022.

## Key Takeaway for the Paper

HiT-IDS occupies a **unique position** in the accuracy–explainability trade-off:

1. **vs. tree-based (RF/XGBoost)**: Trees achieve F1 > 0.999 but require expensive TreeSHAP (272.7 ms/sample) for explanations. HiT-IDS provides intrinsic explanations at 3.43 ms/sample — **79.5× faster**.

2. **vs. Deep-IDS / RTIDS**: LSTM/Transformer baselines achieve competitive F1 but offer **zero intrinsic explainability**. GradientSHAP adds significant per-sample cost.

3. **vs. TranAD**: An unsupervised approach that cannot perform multi-class classification at all. Its reconstruction-error "explanation" is not structured by feature groups.

4. **vs. IDS-INT**: Offers attention visualization but not quantified intrinsic explanation cost. HiT-IDS's hierarchical rollout provides measurable, structured explanations.

The paper's contribution is not topping accuracy — it is providing **intrinsic, hierarchical explanations at negligible latency** while maintaining competitive accuracy.
