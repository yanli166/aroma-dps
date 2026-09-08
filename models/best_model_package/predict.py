"""统一推理接口: 输入 SMILES + 目标环原子索引 -> 同时输出 HOMA / NICS_1zz / MBCO

用法:
    from predict import AromaticityPredictor

    p = AromaticityPredictor('/home/ubuntu/aroma-dps-code/best_model_package')
    res = p.predict('O=C1CNC(=O)N1CCSCC1CCCC1', atom_on_ring=[12, 11, 15, 14, 13])
    # res = {'HOMA': ..., 'NICS_1zz': ..., 'MBCO': ...}

    # 批量
    results = p.predict_batch([(smi1, aor1), (smi2, aor2)])

命令行:
    python predict.py "SMILES" "atom_on_ring(逗号分隔或列表)"
"""
import os
import json
import numpy as np
import torch

from graph_utils import build_graph, NODE_VEC_LEN, MAX_ATOMS
from model_arch import build_model

DEFAULT_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    'HOMA': 'homa_best.pt',
    'NICS_1zz': 'nics_1zz_best.pt',
    'MBCO': 'mbco_best.pt',
}


class AromaticityPredictor:
    def __init__(self, pkg_dir=DEFAULT_PKG_DIR, device=None):
        self.pkg_dir = pkg_dir
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.configs = {}
        for task, fname in MODEL_FILES.items():
            path = os.path.join(pkg_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺少模型权重: {path}")
            ckpt = torch.load(path, map_location='cpu')
            cfg = ckpt['config']
            self.configs[task] = cfg
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
        self.metrics = self._load_metrics()

    def _load_metrics(self):
        path = os.path.join(self.pkg_dir, 'metrics.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    @staticmethod
    def _prep_one(smiles, atom_on_ring, ring_flag):
        g = build_graph(smiles, atom_on_ring, NODE_VEC_LEN, MAX_ATOMS,
                        ring_flag_value=ring_flag)
        node = torch.tensor(g['node_mat'][None], dtype=torch.float32)
        adj = torch.tensor(g['adj_mat'][None], dtype=torch.float32)
        ri = torch.tensor(g['ring_indices'][None], dtype=torch.long)
        return node, adj, ri

    def predict(self, smiles, atom_on_ring, ring_flag=None):
        """预测单分子单环的三任务数值。

        Args:
            smiles: SMILES 字符串
            atom_on_ring: 目标环原子 0-based 索引列表 (与训练数据 CSV 一致)
            ring_flag: 可选, 覆盖配置文件中的 ring_flag_value

        Returns:
            {'HOMA': float, 'NICS_1zz': float, 'MBCO': float}
        """
        out = {}
        for task, model in self.models.items():
            cfg = self.configs[task]
            rf = cfg['ring_flag_value'] if ring_flag is None else ring_flag
            node, adj, ri = self._prep_one(smiles, atom_on_ring, rf)
            with torch.no_grad():
                pred = model(node.to(self.device), adj.to(self.device),
                             ri.to(self.device)).item()
            out[task] = float(pred)
        return out

    def predict_batch(self, items, ring_flag=None):
        """批量预测。items: [(smiles, atom_on_ring), ...] -> list of dict"""
        return [self.predict(smi, aor, ring_flag) for smi, aor in items]


def _parse_cli_aor(s):
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        return eval(s)
    return [int(x) for x in s.replace(',', ' ').split()]


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    smi = sys.argv[1]
    aor = _parse_cli_aor(sys.argv[2])
    p = AromaticityPredictor()
    res = p.predict(smi, aor)
    print(json.dumps(res, indent=2, ensure_ascii=False))
