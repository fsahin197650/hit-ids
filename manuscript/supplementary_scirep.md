# Supplementary information

## Supplementary Table S1. Feature groups (CIC-IDS2017, 61 features)

**Temporal (23)**: Flow Duration, Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min, Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min, Active Mean, Active Std, Active Max, Active Min, Idle Mean, Idle Std, Idle Max, Idle Min, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min

**Volume (24)**: Total Fwd Packets, Total Backward Packets, Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std, Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std, Flow Bytes/s, Flow Packets/s, Fwd Packets/s, Bwd Packets/s, Packet Length Mean, Packet Length Std, Packet Length Variance, Max Packet Length, Min Packet Length, Fwd Header Length, Bwd Header Length, Average Packet Size

**Protocol (10)**: Destination Port, FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, Init_Win_bytes_forward, Init_Win_bytes_backward

**Behavioral (4)**: Subflow Bwd Bytes, Down/Up Ratio, act_data_pkt_fwd, min_seg_size_forward

Removed from the 78 CICFlowMeter columns: constant — Bwd PSH Flags, Bwd URG Flags, Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd Avg Bulk Rate; exact duplicates — Subflow Fwd Packets (= Total Fwd Packets), Subflow Bwd Packets (= Total Backward Packets), Subflow Fwd Bytes (= Total Length of Fwd Packets), Avg Fwd Segment Size (= Fwd Packet Length Mean), Avg Bwd Segment Size (= Bwd Packet Length Mean), Fwd PSH Flags (= SYN Flag Count), Fwd URG Flags (= CWE Flag Count), ECE Flag Count (= RST Flag Count); identifiers — Flow ID, Source IP, Source Port, Destination IP, Timestamp; and the duplicated column Fwd Header Length.1.


## Supplementary Table S2. Hyper-parameters

Token dimension 128; heads 8; intra- and inter-group layers 2 each; feed-forward width 512; dropout 0.1; learning rate 4.54e-4; weight decay 5.17e-5; batch 512; focal γ 4.5; label smoothing 0.1; rollout residual weight 0.5; parameters 2,028,556 (binary) / 2,030,104 (attack-type). Selected by Optuna (100 trials) on the validation split of the deployment model.
