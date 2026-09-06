# Fig.4 lunci10 Naming Protocol (Task 3, by user protocol)

## 禁止使用的术语
- ❌ "completely unseen chemical structures" (lunci10 中 92.7% 有 Murcko scaffold overlap, 97.5% 有 ring-family overlap)
- ❌ "strict scaffold OOD" (真正的 scaffold OOD 来自 internal scaffold split)
- ❌ "lunci10 是 38×30 完整二维矩阵" (实际是 1026/1048 unique combinations, 1389 molecules, 2153 ring-level records)
- ❌ "lunci10 strict external zero-shot scaffold prediction" (因 scaffold/ring-family overlap 高)

## 允许使用的正式术语
- ✅ **"strict molecule-level external benchmark"**
- ✅ **"independently constructed external molecular test set"**
- ✅ **"lunci10 scaffold-substituent-position chemical extrapolation benchmark"**
- ✅ "clean_external" subset (与 internal training 完全无 exact molecule overlap 的部分)
- ✅ "exact_seen" subset (有 overlap, 仅用于 diagnostic)

## 真正 OOD 维度的正确归属
- **strict scaffold OOD**: 由 internal scaffold split 提供
- **strict ring-family OOD**: 由 internal ring-family split 提供
- **lunci10 提供**: 严格 molecule-level external evaluation (确保 exact molecule unseen)

## 主 external analysis 子集
- **唯一使用**: `lunci10_clean_external` (2076 ring-level records / 1331 unique molecules)
- **禁止**: 将 `lunci10_exact_seen` (77 records / 58 molecules) 混入主指标
- **允许**: 在 SI/diagnostic 中单独报告 exact_seen 的性能作为对比