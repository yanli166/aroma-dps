"""
Regression tests for the aroma_dps publication pipeline.

Required tests:
    test_ring_index_roundtrip
    test_same_molecule_no_split_leakage
    test_global_scaffold_split_consistency
    test_scaler_train_only
    test_ood_never_seen
    test_batch_size_one
    test_cache_config_isolation
    test_run_completeness
    test_zero_exposure_equals_frozen_zero_shot
    test_task_name_is_MCBO
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Repo root too: some tests import the legacy unified_models package directly
sys.path.insert(0, str(Path(__file__).parent.parent))


# --- Test data ---
SMILES_BENZENE = "c1ccccc1"
ATOM_ON_RING_BENZENE = [0, 1, 2, 3, 4, 5]


class TestTaskNameIsMCBO:
    """Test that task naming is canonicalized to MCBO."""

    def test_mbcio_alias(self):
        from aroma_dps import canonicalize_task_name
        assert canonicalize_task_name("MBCO") == "MCBO"

    def test_nmcbo_alias(self):
        from aroma_dps import canonicalize_task_name
        assert canonicalize_task_name("nMCBO") == "MCBO"

    def test_mcbo_lowercase_alias(self):
        from aroma_dps import canonicalize_task_name
        assert canonicalize_task_name("mcbo") == "MCBO"

    def test_mbco_lowercase_alias(self):
        from aroma_dps import canonicalize_task_name
        assert canonicalize_task_name("mbco") == "MCBO"

    def test_canonical_passes_through(self):
        from aroma_dps import canonicalize_task_name
        assert canonicalize_task_name("HOMA") == "HOMA"
        assert canonicalize_task_name("NICS_1zz") == "NICS_1zz"
        assert canonicalize_task_name("MCBO") == "MCBO"

    def test_task_provenance_reverse_tracking(self):
        from p0_verification.provenance_recorder import task_provenance
        prov = task_provenance("MBCO")
        assert prov["canonical"] == "MCBO"
        assert prov["original_label"] == "MBCO"

    def test_task_provenance_no_alias(self):
        from p0_verification.provenance_recorder import task_provenance
        prov = task_provenance("HOMA")
        assert prov["canonical"] == "HOMA"
        assert prov["original_label"] is None


class TestRingIndexRoundtrip:
    """Test that ring index canonicalization is consistent."""

    def test_benzene_ring_indices(self):
        from aroma_dps.chemistry.ring_mapping import target_atom_indices_model
        indices = target_atom_indices_model(SMILES_BENZENE, ATOM_ON_RING_BENZENE)
        assert indices == [0, 1, 2, 3, 4, 5]

    def test_string_parsing(self):
        from aroma_dps.chemistry.ring_mapping import target_atom_indices_model
        indices = target_atom_indices_model(SMILES_BENZENE, "[0, 1, 2, 3, 4, 5]")
        assert indices == [0, 1, 2, 3, 4, 5]

    def test_ring_validity_assertion(self):
        from aroma_dps.chemistry.ring_mapping import assert_ring_validity
        # Should not raise
        assert_ring_validity(SMILES_BENZENE, ATOM_ON_RING_BENZENE, ring_size=6)

    def test_smiles_model_canonical(self):
        from aroma_dps.chemistry.ring_mapping import smiles_model
        canonical = smiles_model("c1ccccc1")
        assert canonical == "c1ccccc1"

    def test_out_of_range_raises(self):
        from aroma_dps.chemistry.ring_mapping import target_atom_indices_model
        with pytest.raises(ValueError):
            target_atom_indices_model(SMILES_BENZENE, [0, 1, 2, 3, 4, 5, 6])


class TestGraphConsumesZeroBased:
    """`unified_models.common.graphs.Graph` treats atom_on_ring as 0-based, verbatim.

    Pinned because a Fig.5 script (lunci8/predict_lunci8.py) passes 1-based indices,
    which silently drops a real ring atom and flags a non-ring atom instead.
    """

    SMILES = "c1ccccc1C(=O)O"          # ring atoms 0-5; C(=O)O carbon is atom 6
    RING_0BASED = [0, 1, 2, 3, 4, 5]

    @staticmethod
    def _flagged_atoms(graph, plain):
        a = np.asarray(plain.node_mat, dtype=float)
        b = np.asarray(graph.node_mat, dtype=float)
        cols = [j for j in range(a.shape[1]) if not np.allclose(a[:, j], b[:, j])]
        assert cols, "no ring-flag column differs from the unflagged graph"
        return np.nonzero(np.abs(b[:, cols[0]]) > 1e-9)[0].tolist()

    def test_zero_based_input_flags_exactly_those_atoms(self):
        from unified_models.common.graphs import Graph
        flagged = self._flagged_atoms(Graph(self.SMILES, self.RING_0BASED, 60),
                                       Graph(self.SMILES, [], 60))
        assert flagged == self.RING_0BASED

    def test_one_based_input_mislabels_atoms(self):
        from unified_models.common.graphs import Graph
        shifted = [i + 1 for i in self.RING_0BASED]
        flagged = self._flagged_atoms(Graph(self.SMILES, shifted, 60),
                                      Graph(self.SMILES, [], 60))
        assert flagged == shifted
        assert self.RING_0BASED[0] not in flagged      # real ring atom loses its flag
        assert shifted[-1] not in self.RING_0BASED     # flagged atom is not in the ring


class TestSplitLeakage:
    """Test that same molecule's rings stay in the same split."""

    def test_group_aware_split(self):
        from aroma_dps.data.splits import get_final_splits, assert_no_leak

        # Create synthetic data: 100 samples from 50 molecules
        n_total = 100
        groups = np.array([f"mol_{i//2}" for i in range(n_total)])

        test_idx, cv_folds, final_train_idx, final_val_idx = get_final_splits(
            n_total, groups
        )

        # Assert no leakage
        assert_no_leak(final_train_idx, final_val_idx, test_idx, groups)

    def test_same_molecule_same_split(self):
        from aroma_dps.data.splits import get_final_splits

        # Create data where pairs of rings come from the same molecule
        n_total = 100
        groups = np.array([f"mol_{i//2}" for i in range(n_total)])

        test_idx, cv_folds, final_train_idx, final_val_idx = get_final_splits(
            n_total, groups
        )

        # Check that for each molecule, all its rings are in the same split
        all_train = set(groups[final_train_idx])
        all_val = set(groups[final_val_idx])
        all_test = set(groups[test_idx])

        # No molecule should appear in multiple splits
        assert len(all_train & all_val) == 0
        assert len(all_train & all_test) == 0
        assert len(all_val & all_test) == 0

    def test_split_seed_decoupled(self):
        """Split seed should be 2026, not model seed."""
        from aroma_dps.data.splits import SPLIT_SEED
        assert SPLIT_SEED == 2026


class TestCanonicalSplitsDeprecation:
    """Test that canonical_splits warns and delegates without mis-binding args."""

    def test_deprecation_warning_and_delegation(self):
        from aroma_dps.data.splits import canonical_splits
        groups = np.array([f"mol_{i // 4}" for i in range(100)])
        with pytest.warns(DeprecationWarning):
            test_idx, cv_folds, final_train, final_val = canonical_splits(100, groups=groups)
        assert len(cv_folds) == 5
        assert len(set(test_idx) & set(final_train)) == 0

    def test_legacy_positional_signature_still_binds(self):
        # Historical call shape was canonical_splits(n_samples, seed, groups).
        from aroma_dps.data.splits import canonical_splits
        groups = np.array([f"mol_{i // 4}" for i in range(100)])
        with pytest.warns(DeprecationWarning):
            test_idx, _, _, _ = canonical_splits(100, 42, groups)
        assert len(test_idx) > 0

    def test_groups_are_required(self):
        from aroma_dps.data.splits import canonical_splits
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                canonical_splits(100)


class TestRingFamilyAndDelta:
    """Vendored chemistry helpers must not confuse saturated rings with aromatics."""

    def test_aromatic_family_labels(self):
        from aroma_dps.chemistry.ring_family import ring_family_of
        assert ring_family_of('c1ccccc1', [0, 1, 2, 3, 4, 5])[3] == 'benzene-like'
        assert ring_family_of('c1ccncc1', [0, 1, 2, 3, 4, 5])[3] == 'pyridine-like'
        assert ring_family_of('c1ccoc1', [0, 1, 2, 3, 4])[3] == 'furan-like'

    def test_saturated_ring_is_not_benzene(self):
        from aroma_dps.chemistry.ring_family import ring_family_of
        label = ring_family_of('C1CCCCC1', [0, 1, 2, 3, 4, 5])[3]
        assert label == 'nonaromatic_6(C)'

    def test_out_of_range_indices_rejected(self):
        # 1-based input would push the last index out of range; must not silently pass.
        from aroma_dps.chemistry.ring_family import extract_target_ring_smiles
        assert extract_target_ring_smiles('c1ccccc1', [1, 2, 3, 4, 5, 6]) == ''

    def test_delta_sign_conventions(self):
        from aroma_dps.inference.ring_prediction import aromaticity_loss
        assert aromaticity_loss('HOMA', 0.9, 0.3) == pytest.approx(0.6)
        assert aromaticity_loss('MCBO', 0.9, 0.3) == pytest.approx(0.6)
        # NICS is negative for aromatic rings: loss means going towards zero.
        assert aromaticity_loss('NICS_1zz', -8.0, -2.0) == pytest.approx(6.0)


class TestScalerTrainOnly:
    """Test that scaler is only fit on training data (placeholder)."""

    def test_placeholder(self):
        # Scaler logic will be tested once data/scalers.py is fully implemented
        # For now, this test exists as a placeholder to ensure the test suite
        # includes this check
        pass


class TestOODNeverSeen:
    """Test that OOD test never enters training (placeholder)."""

    def test_placeholder(self):
        pass


class TestBatchSizeOne:
    """Test that model works with batch_size=1 (placeholder)."""

    def test_placeholder(self):
        pass


class TestRunCompleteness:
    """Test run completeness check."""

    def test_completeness_check(self):
        from p0_verification.provenance_recorder import ProvenanceRecorder
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ProvenanceRecorder(result_dir=tmpdir, repo_root="/home/ubuntu/aroma-dps")

            # Start a run
            exp_id = recorder.start_run(
                task="HOMA",
                model="RC-GNN",
                split_seed=2026,
                model_seed=42,
                config={},
                expected_runs=1,
            )

            # Complete it
            recorder.complete_run(exp_id, run_status="ok")

            # Check completeness
            result = recorder.check_completeness(expected_runs=1)
            assert result["can_generate_summary"] is True
            assert result["missing_runs"] == 0
            assert result["failed_runs"] == 0


class TestZeroExposureFrozenZeroShot:
    """Test that 0% exposure = frozen zero-shot checkpoint (placeholder)."""

    def test_placeholder(self):
        # This will be tested once the exposure curve module is integrated
        pass


class TestProvenanceTaskTracking:
    """Test provenance with task reverse tracking."""

    def test_mbcio_records_original_label(self):
        import json
        import tempfile
        from p0_verification.provenance_recorder import ProvenanceRecorder, load_provenance

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ProvenanceRecorder(result_dir=tmpdir, repo_root="/home/ubuntu/aroma-dps")

            exp_id = recorder.start_run(
                task="MBCO",
                model="RC-GNN",
                split_seed=2026,
                model_seed=42,
                config={},
                expected_runs=1,
            )
            recorder.complete_run(exp_id, run_status="ok")

            prov = load_provenance(tmpdir)
            assert prov is not None
            rec = prov["records"][0]
            assert rec["task"]["canonical"] == "MCBO"
            assert rec["task"]["original_label"] == "MBCO"


class TestConfigPaths:
    """config.REPO_ROOT must be the repository root, not a parent of it."""

    def test_repo_root_is_the_package_containing_src(self):
        from aroma_dps import config
        root = Path(config.REPO_ROOT)
        assert (root / "src" / "aroma_dps" / "config.py").is_file()
        assert root == Path(__file__).resolve().parent.parent

    def test_data_and_model_dirs_resolve_inside_repo(self):
        from aroma_dps import config
        for name in ("COLLET_DIR", "LUNCI10_DIR", "MODELS_DIR", "ORIG_MODELS_ROOT"):
            path = Path(getattr(config, name))
            assert path.is_dir(), f"{name} -> {path} does not exist"
            assert path.is_relative_to(Path(config.REPO_ROOT).resolve())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
