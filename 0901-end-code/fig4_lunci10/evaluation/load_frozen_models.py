"""Phase 3 (model loader): 加载冻结的 final RC-GNN 与 Base GNN checkpoints 用于零样本预测.

重要约束 (zero-shot rules):
  - 仅加载 frozen checkpoints, 严禁任何训练 / fine-tune / early-stop
  - 加载训练阶段保存的 scaler (如有), 严禁 fit on lunci10
  - RC-GNN 来自 layer3_ring_fixed (ring_flag=10, label encoding)
  - Base GNN 来自 layer2_gnn (MPNN/GAT/GIN/GNN, label encoding)
  - 任何 scaler / 模型超参均以训练阶段为准, 这里只读取

提供对外接口:
    load_model(task, model_type='RC_GNN', seed=42, encoding='label') -> (model, scaler, meta)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# --------------------------------------------------------------------------
# 把 code_end 加入 sys.path 以复用 unified_models 与 aromatic_split 的训练代码
# --------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# fig4_lunci10/evaluation/ -> fig4_lunci10/ -> 0901-end-code/ -> aroma-dps/
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
CODE_END = os.path.join(PROJ_ROOT, "archive/deprecated/code_end")
ORIG_MODELS_ROOT = os.path.join(PROJ_ROOT, "unified_models")  # unified_models 所在位置

for p in (PROJ_ROOT, CODE_END, ORIG_MODELS_ROOT):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

FIG4_ROOT = Path(PROJ_ROOT) / "0901-end-code/fig4_lunci10"
CONFIG_PATH = FIG4_ROOT / "configs" / "fig4_lunci10.yaml"

# 训练阶段的 checkpoint 根目录 (来自 yaml, 这里 hardcode 以保持独立性)
LAYER2_GNN_DIR = Path(CODE_END) / "results/layer2_gnn"
LAYER3_RING_DIR = Path(CODE_END) / "results/layer3_ring_fixed"
AROMATIC_DIR = Path(CODE_END) / "aromatic_split/results"
SCALER_DIR = Path(CODE_END) / "results/layer2_gnn"  # 训练阶段 scaler 位置

RING_FLAG_VALUE = 10  # 配置中 ring_flag=10 (label 编码)
FEATURE_MODE = "atom_bond"
NVL, MAX_ATOMS = 60, 75  # 与训练阶段一致

# 默认 select_best 策略: 取 layer3_ring_fixed 下 seed_<seed>/<task>/<model>_<encoding>/best_model.pth
# layer3_ring_fixed 包含 5 个 seed 下的多个 encoding (label/mask/pool/combined), 选 label (与 layer2 一致)


def _resolve_checkpoint(task: str, model: str, seed: int, encoding: str = "label",
                       layer: str = "ring_fixed") -> Path:
    """返回 (task, model, seed, encoding, layer) 组合对应的 best_model.pth 路径."""
    if layer == "ring_fixed":
        root = LAYER3_RING_DIR / f"seed_{seed}" / task / f"{model}_{encoding}"
    elif layer == "layer2":
        root = LAYER2_GNN_DIR / f"seed_{seed}" / task / f"{model}_{encoding}"
    else:
        raise ValueError(f"unknown layer: {layer}")
    p = root / "best_model.pth"
    if not p.is_file():
        # fallback: 某些任务 (NICS_1zz) 可能没有 best_model.pth, 提示但仍返回 path
        warnings.warn(f"checkpoint not found: {p}")
    return p


def _resolve_scaler_path(task: str, model: str, seed: int, encoding: str = "label",
                         layer: str = "layer2") -> Optional[Path]:
    """加载训练阶段保存的 feature scaler (e.g. StandardScaler).
    如果 checkpoint 同目录存在 scaler.json / scaler.npy 则优先采用; 否则 None.
    严禁 fit on lunci10 — 只读已存在的训练阶段 artifact.
    """
    if layer == "ring_fixed":
        root = LAYER3_RING_DIR / f"seed_{seed}" / task / f"{model}_{encoding}"
    else:
        root = LAYER2_GNN_DIR / f"seed_{seed}" / task / f"{model}_{encoding}"
    for cand in ("scaler.json", "feature_scaler.json", "scaler.npz"):
        p = root / cand
        if p.is_file():
            return p
    return None


def _load_scaler(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """读取 scaler 系数 (mean/scale 或 a/b), 返回 dict 或 None.
    注意: 此函数禁止任何 fit — 仅读取已存盘的训练阶段 scaler.
    """
    if path is None or not path.is_file():
        return None
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix == ".npz":
        import numpy as np
        npz = np.load(path, allow_pickle=True)
        return {k: npz[k] for k in npz.files}
    return None


def _build_gnn_model(model_name: str, n_outputs: int = 1, ring_flag_value: int = RING_FLAG_VALUE,
                     hidden_dim: int = 128, n_conv: int = 3, n_hidden: int = 2,
                     dropout: float = 0.2, mode: str = "label") -> torch.nn.Module:
    """依据统一模型工厂构造 GNN (layer2/ring_fixed 共享 unified_models).

    Args:
        model_name: GNN/GIN/GAT/MPNN/GraphSAGE
        mode: 'label' / 'mask' / 'pool' / 'combined' / 'none'
    """
    from unified_models.gnn.model import GNNModel
    from unified_models.gin.model import GINModel
    from unified_models.gat.model import GATModel
    from unified_models.mpnn.model import MPNNModel
    from unified_models.graphsage.model import GraphSAGEModel

    registry = {
        "GNN": GNNModel, "GIN": GINModel, "GAT": GATModel,
        "MPNN": MPNNModel, "GraphSAGE": GraphSAGEModel,
    }
    if model_name not in registry:
        raise ValueError(f"unknown base model: {model_name}")

    kwargs = dict(
        node_vec_len=NVL, hidden_dim=hidden_dim, n_conv=n_conv,
        n_hidden=n_hidden, n_outputs=n_outputs, p_dropout=dropout, mode=mode,
    )
    if model_name == "GAT":
        kwargs["n_heads"] = 4
    if model_name in ("GIN", "GAT"):
        kwargs["ring_flag_value"] = ring_flag_value
    return registry[model_name](**kwargs)


def _build_rc_gnn_model(model_name: str, n_outputs: int = 1,
                        ring_flag_value: int = RING_FLAG_VALUE,
                        feature_mode: str = FEATURE_MODE,
                        hidden_dim: int = 128, n_conv: int = 3,
                        n_hidden: int = 2, dropout: float = 0.2,
                        learnable_projection: bool = True,
                        mode: str = "label") -> torch.nn.Module:
    """构造 final RC-GNN: 与 layer3_ring_fixed 完全一致 (ring_flag=10, label 编码,
    learnable projection), 复用 unified_models 的 GNN/GIN/GAT/MPNN.
    """
    return _build_gnn_model(
        model_name=model_name, n_outputs=n_outputs,
        ring_flag_value=ring_flag_value, hidden_dim=hidden_dim,
        n_conv=n_conv, n_hidden=n_hidden, dropout=dropout, mode=mode,
    )


def _safe_torch_load(path: Path, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    """torch.load wrapper — 仅加载不触发任何训练流程."""
    return torch.load(str(path), map_location=map_location, weights_only=False)


def load_model(
    task: str,
    model_type: str = "RC_GNN",
    seed: int = 42,
    encoding: str = "label",
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Optional[Dict[str, Any]], Dict[str, Any]]:
    """加载 frozen 模型 + 训练阶段 scaler.

    Args:
        task: 'HOMA' | 'NICS_1zz' | 'MBCO'
        model_type:
          - 'RC_GNN'  → 来自 layer3_ring_fixed (默认 MPNN label)
          - 'RC_GIN' / 'RC_GAT' / 'RC_GNN_base' / 'RC_MPNN' → layer3_ring_fixed 下对应 backbone
          - 'Base_GNN' / 'Base_GIN' / 'Base_GAT' / 'Base_MPNN' → layer2_gnn
          - 'Aromatic_GNN' / 'Aromatic_MPNN' → aromatic_split/results (all 模式)
        seed: 训练阶段 seed (默认 42; 也支持多 seed ensemble, 但禁止选 seed 用 lunci10)
        encoding: 'label' (默认; 与 layer2 一致)
        device: 'cpu' / 'cuda'

    Returns:
        (model, scaler_dict, meta) 其中 meta 包含 model_type/task/seed/encoding/path/scaler_path.
    """
    if task not in ("HOMA", "NICS_1zz", "MBCO"):
        raise ValueError(f"unknown task: {task}")

    # 解析路径
    if model_type.startswith("RC_"):
        backbone = model_type.replace("RC_", "")  # e.g. GNN, GIN, GAT, MPNN
        backbone_alias = {"GNN_base": "GNN"}.get(backbone, backbone)
        layer = "ring_fixed"
        ckpt = _resolve_checkpoint(task, backbone_alias, seed, encoding, layer=layer)
        model = _build_rc_gnn_model(backbone_alias, mode=encoding)
    elif model_type.startswith("Base_"):
        backbone = model_type.replace("Base_", "")
        backbone_alias = {"GNN_base": "GNN"}.get(backbone, backbone)
        layer = "layer2"
        ckpt = _resolve_checkpoint(task, backbone_alias, seed, encoding, layer=layer)
        model = _build_gnn_model(backbone_alias, mode=encoding)
    elif model_type.startswith("Aromatic_"):
        backbone = model_type.replace("Aromatic_", "")
        ckpt = AROMATIC_DIR / task / "all" / backbone / "best_model.pth"
        model = _build_gnn_model(backbone, mode=encoding)
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    # 1) 加载 frozen weights — 严格 torch.load, 不进行任何 optimizer / scheduler 操作
    if ckpt.is_file():
        state = _safe_torch_load(ckpt, map_location=device)
        # 兼容 {"state_dict": ..., "model": ...} 两种格式
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state and isinstance(state["model"], dict):
                state = state["model"]
        model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 2) 加载训练阶段 scaler (feature scaler), 严禁 fit on lunci10
    layer_for_scaler = "ring_fixed" if model_type.startswith("RC_") else (
        "ring_fixed" if model_type.startswith("Aromatic_") else "layer2"
    )
    backbone_alias = model_type.split("_", 1)[1] if "_" in model_type else model_type
    scaler_path = _resolve_scaler_path(task, backbone_alias, seed, encoding,
                                       layer=layer_for_scaler)
    scaler = _load_scaler(scaler_path)

    meta = {
        "model_type": model_type,
        "task": task,
        "seed": seed,
        "encoding": encoding,
        "ring_flag_value": RING_FLAG_VALUE,
        "feature_mode": FEATURE_MODE,
        "learnable_projection": True,  # layer3 ring_fixed 启用 learnable projection
        "checkpoint_path": str(ckpt),
        "scaler_path": str(scaler_path) if scaler_path else None,
        "nvl": NVL,
        "max_atoms": MAX_ATOMS,
    }
    return model, scaler, meta


def list_available_checkpoints(task: str = "HOMA") -> Dict[str, Any]:
    """列出可用 (seed × model × encoding × layer) checkpoint 组合, 用于诊断."""
    out: Dict[str, Any] = {}
    for seed in (42, 123, 456, 789, 2024):
        for layer, root in (("ring_fixed", LAYER3_RING_DIR),
                            ("layer2", LAYER2_GNN_DIR)):
            base = root / f"seed_{seed}" / task
            if not base.is_dir():
                continue
            for sub in sorted(base.iterdir()):
                if (sub / "best_model.pth").is_file():
                    out.setdefault((seed, layer), []).append(sub.name)
    return out


if __name__ == "__main__":
    # 诊断入口: 仅列出可用 checkpoint, 不加载模型
    for t in ("HOMA", "NICS_1zz", "MBCO"):
        print(f"=== {t} ===")
        avail = list_available_checkpoints(t)
        for (seed, layer), models in sorted(avail.items()):
            print(f"  seed={seed} layer={layer}: {len(models)} entries")
            for m in models[:5]:
                print(f"    - {m}")
            if len(models) > 5:
                print(f"    ... (+{len(models) - 5} more)")
