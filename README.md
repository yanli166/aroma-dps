# Aroma-DPS: Deep Learning for Aromaticity Prediction

Predicting ring-level aromaticity indicators (HOMA, NICS(1)zz, MBCO) using Graph Neural Networks with target-ring conditioning.

## Project Structure

```
aroma-dps/
├── 0831-end-code/          # Fig.3: Ring-conditioned GNN ablation pipeline
│   ├── common/             # Shared modules (constants, tasks, protocol, train_eval)
│   ├── models/              # Model definitions (ring-conditioned GNN, PyG models, ring readout)
│   ├── stage1_representation_comparison/   # Stage 1: ML vs GNN baselines
│   ├── stage2_ring_conditioning/           # Stage 2: 2×2 ablation (ring_flag × readout)
│   ├── stage3_mask_pretraining/             # Stage 3: Ring masking pretraining
│   ├── stage4_cross_architecture/          # Stage 4: Cross-architecture validation
│   ├── stage5_ring_flag_sensitivity/       # Stage 5: ring_flag amplitude sensitivity
│   ├── stage6_final_membership/             # Stage 6: Final membership scheme
│   ├── multiseed/          # Multi-seed experiment runner
│   ├── stats/              # Statistical significance tests
│   ├── splits/             # Data split manifests (JSON)
│   ├── run_all_v2_parallel.sh   # Main entry: 5 seeds × 2 feature modes × 6 stages
│   └── PROTOCOL_SPEC_FINAL.md   # Protocol specification
│
├── 0901-end-code/          # Fig.4: Generalization to unseen scaffolds
│   └── fig4_lunci10/       # lunci10 external benchmark
│       ├── configs/        # YAML configuration
│       ├── data/           # Data preparation scripts
│       ├── audit/          # Overlap and provenance auditing
│       ├── evaluation/     # Zero-shot external prediction
│       ├── training/       # Scaffold OOD, exposure curve, lunci10 adaptation
│       ├── analysis/       # Novelty, scaffold bias, Hammett analysis
│       └── reporting/      # Report generation and run registry
│
├── code_end/               # Shared infrastructure
│   ├── common/             # Constants, tasks, graph data, features
│   ├── ring_encoding_ablation/   # Ring encoding training/eval
│   ├── baseline_gnn/       # Baseline GNN models
│   ├── baseline_traditional_ml/  # Traditional ML baselines
│   ├── layer4_substituent/ # Substituent-level models (Hammett, attention, etc.)
│   ├── generalization_test/ # OOD generalization benchmarks
│   ├── aromatic_split/    # Aromaticity-based split analysis
│   ├── optuna_search/     # Hyperparameter optimization
│   └── stats/             # Significance testing
│
└── unified_models/         # Core model definitions
    ├── common/             # Graph processing (graphs.py), utilities
    ├── gnn/                # GNN model
    ├── mpnn/               # Message Passing Neural Network
    ├── gat/                # Graph Attention Network
    ├── gin/                # Graph Isomorphism Network
    └── graphsage/          # GraphSAGE model
```

## Key Features

- **Ring-Conditioned GNN**: Target ring atoms are explicitly marked in node features, enabling ring-level predictions
- **E*=median Protocol**: Fixed 80/20 holdout → 5-fold Group CV → E* = median(best epochs) → full dev retrain
- **Dual Feature Mode**: `standard` (full features) vs `explicit_aromaticity_ablated` (remove handcrafted aromaticity flags)
- **Scaffold OOD Generalization**: Internal scaffold split + lunci10 external benchmark with exposure curves
- **Three Aromaticity Tasks**: HOMA (geometric), NICS(1)zz (magnetic), MBCO (bond order)

## Requirements

- Python 3.10+
- PyTorch
- PyTorch Geometric (PyG)
- RDKit
- scikit-learn
- pandas, numpy

## Usage

### Fig.3 Reproduction (0831-end-code)

```bash
cd 0831-end-code
bash run_all_v2_parallel.sh
```

Runs 5 seeds (11, 22, 33, 44, 55) × 2 feature modes × 6 stages on 3 GPUs.

### Fig.4 Generalization (0901-end-code)

```bash
cd 0901-end-code/fig4_lunci10

# Experiment A: Internal Scaffold OOD
python training/train_scaffold_ood.py

# Experiment B: Internal Exposure Curve
python training/train_exposure_curve.py

# Experiment C: lunci10 Scaffold Exposure
python training/train_l10_exposure.py

# Phase 4: Zero-shot External Prediction
python evaluation/run_external_absolute_v2.py
```

## Data

Training data: collet_homa_0716.csv, collet_nics_0716.csv, collet_mbco_0716.csv
External benchmark: lunci10-test-corrected.csv (1389 molecules, 2153 ring-level records)

## License

Research use only.
