"""
Split logic: single source of truth.

Publication protocol uses get_final_splits(SPLIT_SEED=2026).
canonical_splits is kept as a deprecated wrapper for backward compatibility
with historical checkpoint/result loaders that may depend on the import path.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# --- Publication split seed ---
SPLIT_SEED = 2026


def get_final_splits(
    n_total: int,
    groups: np.ndarray,
    test_size: float = 0.2,
    n_folds: int = 5,
    split_seed: int = SPLIT_SEED,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Publication split: fixed test set + GroupKFold CV.

    Guarantees:
    - Same molecule's all target rings are in the same split.
    - SPLIT_SEED=2026 is decoupled from model seed.
    - Test set never participates in training/validation/early stopping/
      architecture selection/checkpoint selection/normalization fitting.

    Args:
        n_total: Total number of ring-level samples.
        groups: Canonical SMILES for each sample (for group-aware splitting).
        test_size: Fraction of groups held out for test.
        n_folds: Number of CV folds within the development set.
        split_seed: Fixed split seed (default 2026).

    Returns:
        test_idx: Test set indices.
        cv_folds: List of (train_idx, val_idx) tuples.
        final_train_idx: Final training indices (all development data).
        final_val_idx: Final validation indices (held out for early stopping).
    """
    rng = np.random.RandomState(split_seed)

    # --- Step 1: Group-aware test split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=split_seed)
    dev_idx, test_idx = next(gss.split(np.arange(n_total), groups=groups))

    # --- Step 2: GroupKFold CV within development set ---
    dev_groups = groups[dev_idx]
    gkf = GroupKFold(n_splits=n_folds)
    cv_folds = []
    for train_local, val_local in gkf.split(dev_idx, groups=dev_groups):
        train_idx = dev_idx[train_local]
        val_idx = dev_idx[val_local]
        cv_folds.append((train_idx, val_idx))

    # --- Step 3: Final train/val split for full development training ---
    gss_final = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=split_seed)
    final_train_local, final_val_local = next(
        gss_final.split(dev_idx, groups=dev_groups)
    )
    final_train_idx = dev_idx[final_train_local]
    final_val_idx = dev_idx[final_val_local]

    # --- Assert no leakage ---
    assert_no_leak(train_idx=final_train_idx, val_idx=final_val_idx,
                   test_idx=test_idx, groups=groups)

    return test_idx, cv_folds, final_train_idx, final_val_idx


def assert_no_leak(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: np.ndarray,
):
    """Assert no molecule appears in multiple splits."""
    train_smiles = set(groups[train_idx])
    val_smiles = set(groups[val_idx])
    test_smiles = set(groups[test_idx])

    # Check pairwise disjoint
    assert len(train_smiles & val_smiles) == 0, (
        f"Leakage: {len(train_smiles & val_smiles)} molecules in both train and val"
    )
    assert len(train_smiles & test_smiles) == 0, (
        f"Leakage: {len(train_smiles & test_smiles)} molecules in both train and test"
    )
    assert len(val_smiles & test_smiles) == 0, (
        f"Leakage: {len(val_smiles & test_smiles)} molecules in both val and test"
    )


def canonical_splits(n_samples, seed=42, groups=None, test_size=0.2, n_folds=5):
    """DEPRECATED: Use get_final_splits() instead.

    Keeps the historical call shape (n_samples, seed, groups) and translates it
    onto the publication API. The split is NOT identical to the historical one
    (this protocol uses a fixed 80/20 group holdout, GroupKFold within dev and a
    90/10 final train/val cut), but argument binding is now explicit, so a
    misplaced seed fails loudly instead of silently becoming `groups`.

    Retained for 6-12 months, then will be removed.
    """
    warnings.warn(
        "canonical_splits() is deprecated. Use get_final_splits() instead. "
        "SPLIT_SEED=2026 is decoupled from model seed for publication protocol.",
        DeprecationWarning,
        stacklevel=2,
    )
    if groups is None:
        raise ValueError(
            "get_final_splits requires group-aware splitting (parent-molecule SMILES); "
            "the legacy sample-level fallback is not publication-safe."
        )
    return get_final_splits(
        n_total=n_samples,
        groups=groups,
        test_size=test_size,
        n_folds=n_folds,
        split_seed=seed,
    )
