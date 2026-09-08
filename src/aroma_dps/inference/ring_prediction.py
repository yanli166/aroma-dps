"""
Ring prediction: target-ring aromaticity inference and aromaticity-loss (ΔA).

Wraps the released three-task checkpoints in models/best_model_package/
(AromaticityPredictor's logic, re-expressed against the publication layout)
and centralises the ΔA sign conventions used by Fig.4f / Fig.5 / Fig.6:

    ΔHOMA  = HOMA_R  - HOMA_P     (aromaticity loss > 0)
    ΔMCBO  = MCBO_R  - MCBO_P     (aromaticity loss > 0)
    ΔNICS* = NICS_P  - NICS_R     (NICS is negative for aromatic rings,
                                   so the difference is reversed)

ΔA > 0 always means the product ring lost aromaticity relative to the reactant.

Task naming: the released checkpoints are keyed HOMA / NICS_1zz / MBCO, while
the publication protocol canonicalises the bond-order indicator as MCBO. Both
spellings are accepted here and normalised to the checkpoint keys.
"""
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from aroma_dps import config
from aroma_dps.chemistry.ring_mapping import target_atom_indices_model

DEFAULT_PKG_DIR = os.path.join(config.MODELS_DIR, 'best_model_package')

CHECKPOINT_FILES = {
    'HOMA': 'homa_best.pt',
    'NICS_1zz': 'nics_1zz_best.pt',
    'MBCO': 'mbco_best.pt',
}

_ALIASES = {
    'homa': 'HOMA',
    'nics': 'NICS_1zz',
    'nics_1zz': 'NICS_1zz',
    'nics(1)zz': 'NICS_1zz',
    'mbco': 'MBCO',
    'mcbo': 'MBCO',
}


def normalize_task(name: str) -> str:
    """Map any task spelling to the checkpoint key ('MBCO' == MCBO)."""
    key = str(name).strip().lower().replace(' ', '')
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"Unknown task: {name}. Available: {sorted(CHECKPOINT_FILES)}")


def aromaticity_loss(task: str, reactant: float, product: float) -> float:
    """Signed aromaticity loss of a reaction: > 0 means dearomatization."""
    t = normalize_task(task)
    if t == 'NICS_1zz':
        return product - reactant
    return reactant - product


def aromaticity_gain(task: str, reactant: float, product: float) -> float:
    """Signed re-aromatization gain: > 0 means the product ring is more aromatic."""
    return -aromaticity_loss(task, reactant, product)


class RingAromaticityPredictor:
    """Load the released checkpoints and predict ring-level aromaticity.

    Args:
        pkg_dir: directory holding model_arch.py, graph_utils.py and the .pt
                 files (defaults to models/best_model_package).
        device:  torch device string; defaults to cuda:0 when available.
        tasks:   subset of ('HOMA', 'NICS_1zz', 'MBCO') to load.
    """

    def __init__(self, pkg_dir: str = DEFAULT_PKG_DIR,
                 device: Optional[str] = None,
                 tasks: Optional[Iterable[str]] = None):
        import torch

        self.torch = torch
        self.pkg_dir = pkg_dir
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        from graph_utils import MAX_ATOMS, NODE_VEC_LEN, build_graph
        from model_arch import build_model

        self.build_graph = build_graph
        self.node_vec_len = NODE_VEC_LEN
        self.max_atoms = MAX_ATOMS
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

        wanted = [normalize_task(t) for t in (tasks or CHECKPOINT_FILES)]
        self.models: Dict[str, object] = {}
        self.configs: Dict[str, dict] = {}
        for task in wanted:
            path = os.path.join(pkg_dir, CHECKPOINT_FILES[task])
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing checkpoint: {path}")
            ckpt = torch.load(path, map_location='cpu')
            cfg = ckpt['config']
            model = build_model(
                use_projection=cfg['use_projection'],
                node_vec_len=cfg['node_vec_len'],
                hidden_dim=cfg['hidden_dim'],
                n_conv=cfg['n_conv'],
                n_hidden=cfg['n_hidden'],
                p_dropout=cfg['p_dropout'],
                ring_flag_value=cfg['ring_flag_value'],
            )
            model.load_state_dict(ckpt['state_dict'])
            model.to(self.device)
            model.eval()
            self.models[task] = model
            self.configs[task] = cfg

    def _tensors(self, smiles: str, atom_on_ring: Union[str, List[int]], ring_flag: int):
        torch = self.torch
        indices = target_atom_indices_model(smiles, atom_on_ring)
        graph = self.build_graph(smiles, indices, self.node_vec_len, self.max_atoms,
                                 ring_flag_value=ring_flag)
        return (
            torch.tensor(graph['node_mat'][None], dtype=torch.float32),
            torch.tensor(graph['adj_mat'][None], dtype=torch.float32),
            torch.tensor(graph['ring_indices'][None], dtype=torch.long),
        )

    def predict(self, smiles: str, atom_on_ring: Union[str, List[int]],
                ring_flag: Optional[int] = None) -> Dict[str, float]:
        """Predict all loaded tasks for one (molecule, target ring) record.

        atom_on_ring must be 0-based indices on the hydrogen-added molecule,
        matching the training CSVs (verified by the P0-2 audit).
        """
        torch = self.torch
        out: Dict[str, float] = {}
        for task, model in self.models.items():
            rf = self.configs[task]['ring_flag_value'] if ring_flag is None else ring_flag
            node, adj, ring_idx = self._tensors(smiles, atom_on_ring, rf)
            with torch.no_grad():
                value = model(node.to(self.device), adj.to(self.device),
                              ring_idx.to(self.device)).item()
            out[task] = float(value)
        return out

    def predict_batch(self, items: Sequence[Tuple[str, Union[str, List[int]]]],
                      ring_flag: Optional[int] = None) -> List[Dict[str, float]]:
        return [self.predict(smiles, indices, ring_flag) for smiles, indices in items]

    def predict_reaction_pairs(
        self,
        pairs: Sequence[Tuple[Tuple[str, Union[str, List[int]]],
                              Tuple[str, Union[str, List[int]]]]],
        ring_flag: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Predict reactant/product rings together and add ΔA per task.

        Args:
            pairs: ((reactant_smiles, reactant_ring), (product_smiles, product_ring))
        Returns:
            One dict per pair: {task}_reactant, {task}_product, delta_{task}.
        """
        results = []
        for (r_smi, r_ring), (p_smi, p_ring) in pairs:
            reactant = self.predict(r_smi, r_ring, ring_flag)
            product = self.predict(p_smi, p_ring, ring_flag)
            row = {f'{t}_reactant': v for t, v in reactant.items()}
            row.update({f'{t}_product': v for t, v in product.items()})
            row.update({f'delta_{t}': aromaticity_loss(t, reactant[t], product[t])
                        for t in reactant})
            results.append(row)
        return results
