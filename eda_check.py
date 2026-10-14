import csv
from collections import defaultdict, Counter

csv_path = "data/z_alizadeh_sani_extended.csv"

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)
    rows = list(reader)

print(f"Total headers: {len(headers)}, Total records: {len(rows)}")

col_values = defaultdict(list)
for r in rows:
    for h, val in zip(headers, r):
        col_values[h].append(val)

num_cols = []
cat_cols = []

for h in headers:
    vals = col_values[h]
    # Check if all can be float
    is_num = True
    for v in vals:
        try:
            float(v)
        except ValueError:
            is_num = False
            break
    if is_num:
        num_cols.append(h)
    else:
        cat_cols.append(h)

print(f"\nNumeric columns ({len(num_cols)}):")
for col in num_cols:
    vals = [float(x) for x in col_values[col]]
    print(f"  {col}: min={min(vals)}, max={max(vals)}, sample={vals[:3]}")

print(f"\nCategorical columns ({len(cat_cols)}):")
for col in cat_cols:
    c = Counter(col_values[col])
    print(f"  {col}: {dict(c)}")
