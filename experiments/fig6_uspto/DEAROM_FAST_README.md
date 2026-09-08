# Fast dearomatization prescreener for USPTO_STEREO

## Why this version
The original full-corpus workflow runs FMCS for every reaction and keeps results in memory until the end. That is unsuitable for ~1M reactions. This version:

1. streams the input file line by line;
2. accepts a one-column `reactions` CSV directly;
3. uses atom maps when present;
4. otherwise tracks an aromatic target ring with element-specific ring SMARTS and any bond order (`~`), so aromatic -> single/double remains matchable;
5. uses no global FMCS in the fast pass;
6. writes P1/P2/P3 candidates incrementally;
7. supports multiprocessing and `--start-row` / `--limit`.

## Priority meaning
- **P1**: retained target-ring topology + substantial local aromaticity loss + local reaction-center evidence + sufficient ring-context retention.
- **P2**: retained target ring with partial/weaker aromaticity loss evidence.
- **P3**: global aromaticity decreases, but no retained dearomatized target ring was established. This pool intentionally contains fragment loss, deprotection, ring opening, and mapping failures and requires review.
- **P0**: no structural dearomatization evidence in the fast pass.

These are screening priorities, not mechanistic truth labels.

## Input integrity check
Do not use `sep=None` on the one-column header `reactions`; delimiter sniffing can mis-detect the `t` in `reactions` as a separator.

```bash
python - <<'PY'
import re
p='/home/ubuntu/aroma-dps-code/uspto-5k/USPTO_STEREO.csv'
with open(p, encoding='utf-8', errors='replace') as f:
    print('header:', repr(f.readline().rstrip()))
    mapped = 0
    valid = 0
    for i in range(1000):
        line=f.readline().rstrip('\r\n')
        if not line: break
        valid += (line.count('>') == 2)
        mapped += bool(re.search(r':\d+\]', line))
        if i < 3:
            print(i, 'gt=', line.count('>'), 'len=', len(line), line[:220])
    print('valid reaction rows among first 1000:', valid)
    print('rows with atom-map syntax among first 1000:', mapped)
PY
```

## Pilot
```bash
cd /home/ubuntu/aroma-dps-code/uspto-5k
python dearom_screen_fast.py \
  --input USPTO_STEREO.csv \
  --outdir dearom_stereo_fast_10k \
  --workers 16 \
  --chunksize 256 \
  --limit 10000
```

Inspect:
```bash
cat dearom_stereo_fast_10k/screening_report.json
wc -l dearom_stereo_fast_10k/priority_1.csv
wc -l dearom_stereo_fast_10k/priority_2.csv
wc -l dearom_stereo_fast_10k/priority_3.csv
```

## Full pass
```bash
cd /home/ubuntu/aroma-dps-code/uspto-5k
python dearom_screen_fast.py \
  --input USPTO_STEREO.csv \
  --outdir dearom_stereo_fast_full \
  --workers 24 \
  --chunksize 512 \
  --checkpoint-every 10000
```

By default P0 rows are not written, which keeps output smaller. Add `--write-all` only if you really need every P0 reaction row.

## Recommended second stage
Do **not** run FMCS over the full USPTO_STEREO file. After the fast pass:

1. take P1 + P2 candidates;
2. atom-map them with RXNMapper if they are unmapped;
3. re-run exact mapped-ring validation;
4. manually label stratified samples from P1/P2/P3 to calibrate precision;
5. then build DearomCorpus-Silver/Gold.
