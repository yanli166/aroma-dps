#!/usr/bin/env python3
r"""NICS Pipeline 配置注册表 (config-driven)

新增数据集: 在 DATASETS 中追加一条目即可, 无需改动任何脚本。

每个数据集必填字段:
    input_dir   - 含 .chk/.gjf/.log (优化计算结果) 的目录
    csv_path    - SMILES CSV (必须含 'no' 和 'SMILES' 列)
    out_dir     - 输出根目录 (自动创建)

可选字段 (有默认值):
    opt_log_pattern  - 优化log文件名正则, group(1)=mol_id   默认 '^(.+)\.log$'
    opt_log_prefix   - 从opt log提取mol_id后追加的前缀       默认 '' (lunci6用'e')
    nics_log_pattern - NMR NICS log正则, group(1)=mol_id     默认 '^(.+)-nics\.log$'
    multiplicity     - Gaussian 电荷与多重度                  默认 '0 1' (单重态)
    nics_title       - NICS GJF 标题                          默认 'NICS calculation S'
    nics_log_dir     - NMR log所在目录 (默认 out_dir/nics_gjf)
    fchk_search_dirs - 空格分隔的多目录, 用于查找fchk (默认 out_dir/fchk)

派生路径 (由 out_dir 自动推导, 无需配置):
    ring_info_csv, homa_mbco_csv, final_csv,
    homa_txt_dir, mbco_txt_dir, nics_gjf_dir, fchk_dir,
    nics_multiwfn_dir, multiwfn_input_dir, multiwfn_output_dir,
    outhoma_dir, outmbco_dir

CLI用法 (shell脚本调用):
    python3 config.py                       # 列出已注册数据集
    python3 config.py datasets              # 同上, 空格分隔
    python3 config.py <dataset> <key>       # 打印单个配置值
    python3 config.py --shell <dataset>     # 打印 shell-sourceable 的变量赋值
"""
import os
import sys
import shlex

DEFAULTS = {
    "opt_log_pattern": r'^(.+)\.log$',
    "opt_log_prefix": "",
    "nics_log_pattern": r'^(.+)-nics\.log$',
    "multiplicity": "0 1",
    "nics_title": "NICS calculation S",
}

DATASETS = {
    "lunci7": {
        "input_dir": "/home/ubuntu/data_90/alldata_in_3090/cal/DPSCAL/lunci7/s1",
        "csv_path": "/home/ubuntu/data_90/alldata_in_3090/cal/DPSCAL/lunci7/lunci7-begin.csv",
        "out_dir": "/home/ubuntu/aroma-dps-code/lunci7_8_out_nics/lunci7",
    },
    "lunci8": {
        "input_dir": "/home/ubuntu/aroma-dps-code/lunci8/gaussian_inputs",
        "csv_path": "/home/ubuntu/aroma-dps-code/lunci8/lunci8-begin.csv",
        "out_dir": "/home/ubuntu/aroma-dps-code/lunci7_8_out_nics/lunci8",
    },
    "lunci8_2": {
        "input_dir": "/home/ubuntu/aroma-dps-code/lunci8/gaussian_inputs",
        "csv_path": "/home/ubuntu/aroma-dps-code/last_end_code/lunci8-begin-2.csv",
        "out_dir": "/home/ubuntu/aroma-dps-code/lunci7_8_out_nics/lunci8_2",
    },
    "lunci8_3": {
        "input_dir": "/home/ubuntu/aroma-dps-code/lunci8-3/gaussian_inputs",
        "csv_path": "/home/ubuntu/aroma-dps-code/lunci8-3/lunci8-3-begin.csv",
        "out_dir": "/home/ubuntu/aroma-dps-code/lunci7_8_out_nics/lunci8-3",
    },
    "lunci9": {
        "input_dir": "/home/ubuntu/aroma-dps-code/lunci9/gaussian_inputs",
        "csv_path": "/home/ubuntu/aroma-dps-code/lunci9/lunci9-begin.csv",
        "out_dir": "/home/ubuntu/aroma-dps-code/lunci9_out",
    },
}

DERIVED = {
    "ring_info_csv":       lambda d, ds: os.path.join(d["out_dir"], "ring_info.csv"),
    "homa_mbco_csv":       lambda d, ds: os.path.join(d["out_dir"], f"{ds}-homa-mbco-summary.csv"),
    "final_csv":           lambda d, ds: os.path.join(d["out_dir"], f"{ds}-homa-mbco-nics-final.csv"),
    "homa_txt_dir":        lambda d, ds: os.path.join(d["out_dir"], "homatxt"),
    "mbco_txt_dir":        lambda d, ds: os.path.join(d["out_dir"], "mbcotxt"),
    "nics_gjf_dir":        lambda d, ds: os.path.join(d["out_dir"], "nics_gjf"),
    "fchk_dir":            lambda d, ds: os.path.join(d["out_dir"], "fchk"),
    "nics_log_dir":        lambda d, ds: d.get("nics_log_dir", os.path.join(d["out_dir"], "nics_gjf")),
    "fchk_search_dirs":    lambda d, ds: d.get("fchk_search_dirs", os.path.join(d["out_dir"], "fchk")),
    "nics_multiwfn_dir":   lambda d, ds: os.path.join(d["out_dir"], "nics_multiwfn"),
    "multiwfn_input_dir":  lambda d, ds: os.path.join(d["out_dir"], "nics_multiwfn", "multiwfn_input"),
    "multiwfn_output_dir": lambda d, ds: os.path.join(d["out_dir"], "nics_multiwfn", "multiwfn_output"),
    "outhoma_dir":         lambda d, ds: os.path.join(d["out_dir"], "OUTHOMA"),
    "outmbco_dir":         lambda d, ds: os.path.join(d["out_dir"], "OUTMBCO"),
}

ALL_KEYS = ["input_dir", "csv_path", "out_dir",
            "opt_log_pattern", "opt_log_prefix", "nics_log_pattern",
            "multiplicity", "nics_title",
            "ring_info_csv", "homa_mbco_csv", "final_csv",
            "homa_txt_dir", "mbco_txt_dir", "nics_gjf_dir", "fchk_dir",
            "nics_log_dir", "fchk_search_dirs",
            "nics_multiwfn_dir", "multiwfn_input_dir", "multiwfn_output_dir",
            "outhoma_dir", "outmbco_dir"]


def get(dataset, key):
    if dataset not in DATASETS:
        raise KeyError(f"未知数据集: {dataset}. 已注册: {list(DATASETS.keys())}")
    d = DATASETS[dataset]
    if key in d:
        return d[key]
    if key in DERIVED:
        merged = dict(DEFAULTS)
        merged.update(d)
        return DERIVED[key](merged, dataset)
    if key in DEFAULTS:
        return DEFAULTS[key]
    raise KeyError(f"未知配置项: {key}")


def list_datasets():
    return list(DATASETS.keys())


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("已注册数据集: " + ", ".join(list_datasets()))
        sys.exit(0)
    if sys.argv[1] == "datasets":
        print(" ".join(list_datasets()))
        sys.exit(0)
    if sys.argv[1] == "--shell":
        ds = sys.argv[2]
        for k in ALL_KEYS:
            print(f'{k.upper()}={shlex.quote(str(get(ds, k)))}')
        sys.exit(0)
    if len(sys.argv) == 2:
        ds = sys.argv[1]
        for k in ALL_KEYS:
            print(f"{k} = {get(ds, k)}")
        sys.exit(0)
    ds, key = sys.argv[1], sys.argv[2]
    print(get(ds, key))
