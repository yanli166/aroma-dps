
# --- Auto path bootstrap (do not remove) ---
import os as _os
_THIS_FILE = _os.path.abspath(__file__)
_d = _os.path.dirname(_THIS_FILE)
while not _os.path.exists(_os.path.join(_d, 'unified_models')) and _d != '/':
    _d = _os.path.dirname(_d)
_PROJ_ROOT = _d
# --- End auto path bootstrap ---

#!/usr/bin/env python3
"""
Anchor-based Δ-Learning Step 7: Anchor robustness 对比 all-pairs
"""
import os
import pandas as pd
import numpy as np

OUTPUT_DIR = '_PROJ_ROOT + "/code_end"/results/lunci10_anchor_delta_final'
ALL_PAIRS_DIR = '_PROJ_ROOT + "/code_end"/results/lunci10_delta_learning'

def main():
    df_anchor = pd.read_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'))

    # All-pairs results
    df_sub_all = pd.read_csv(os.path.join(ALL_PAIRS_DIR, '02_subtraction_baseline/subtraction_agg_all.csv'))
    df_sia_all = pd.read_csv(os.path.join(ALL_PAIRS_DIR, '04_siamese_mpnn/siamese_agg_all.csv'))
    df_fp_all = pd.read_csv(os.path.join(ALL_PAIRS_DIR, '03_delta_fingerprint/delta_fp_agg.csv'))

    rows = []

    for task in ['HOMA', 'NICS_1zz', 'MBCO']:
        # Anchor-based (跨 anchor 平均)
        sub = df_anchor[df_anchor['task'] == task]
        anchor_avg = sub.groupby('model').agg({
            'micro_r2_mean': 'mean', 'micro_mae_mean': 'mean',
            'micro_spearman_mean': 'mean', 'micro_pairwise_acc_mean': 'mean',
            'macro_r2_mean': 'mean', 'macro_mae_mean': 'mean',
            'macro_spearman_mean': 'mean', 'macro_pairwise_acc_mean': 'mean',
        }).reset_index()

        for _, r in anchor_avg.iterrows():
            rows.append({
                'task': task, 'model': r['model'], 'source': 'anchor-based (3 anchors avg)',
                'micro_r2': r['micro_r2_mean'], 'micro_mae': r['micro_mae_mean'],
                'micro_spearman': r['micro_spearman_mean'], 'micro_pairwise': r['micro_pairwise_acc_mean'],
                'macro_r2': r['macro_r2_mean'], 'macro_mae': r['macro_mae_mean'],
                'macro_spearman': r['macro_spearman_mean'], 'macro_pairwise': r['macro_pairwise_acc_mean'],
            })

        # Anchor-based per anchor (展示变异性)
        for anchor in ['F', 'Cl', 'OMe']:
            sub_a = sub[sub['anchor'] == anchor]
            for _, r in sub_a.iterrows():
                rows.append({
                    'task': task, 'model': r['model'], 'source': f'anchor={anchor}',
                    'micro_r2': r['micro_r2_mean'], 'micro_mae': r['micro_mae_mean'],
                    'micro_spearman': r['micro_spearman_mean'], 'micro_pairwise': r['micro_pairwise_acc_mean'],
                    'macro_r2': r['macro_r2_mean'], 'macro_mae': r['macro_mae_mean'],
                    'macro_spearman': r['macro_spearman_mean'], 'macro_pairwise': r['macro_pairwise_acc_mean'],
                })

        # All-pairs
        for model_name, df_all in [('subtraction', df_sub_all), ('siamese', df_sia_all), ('deltafp', df_fp_all)]:
            if model_name == 'deltafp':
                r = df_all[(df_all['task'] == task) & (df_all['model'] == 'XGBoost')]
            else:
                r = df_all[df_all['task'] == task]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            rows.append({
                'task': task, 'model': model_name, 'source': 'all-pairs',
                'micro_r2': r['r2_mean'], 'micro_mae': r['mae_mean'],
                'micro_spearman': r['spearman_mean'], 'micro_pairwise': r.get('pairwise_mean', np.nan),
                'macro_r2': np.nan, 'macro_mae': np.nan,
                'macro_spearman': np.nan, 'macro_pairwise': np.nan,
            })

    df_rob = pd.DataFrame(rows)
    df_rob.to_csv(os.path.join(OUTPUT_DIR, 'anchor_robustness.csv'), index=False)
    print("=== Anchor Robustness: Anchor-based vs All-pairs ===")
    print(df_rob.to_string(index=False, float_format='%.4f'))
    print(f"\nSaved: {OUTPUT_DIR}/anchor_robustness.csv")

if __name__ == '__main__':
    main()
