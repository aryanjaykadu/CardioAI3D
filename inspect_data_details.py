import zipfile
import xml.etree.ElementTree as ET
from collections import Counter

xlsx_path = "data/extention of Z-Alizadeh sani dataset.xlsx"

with zipfile.ZipFile(xlsx_path, 'r') as z:
    shared_strings = []
    if 'xl/sharedStrings.xml' in z.namelist():
        tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
        for si in tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
            t_elem = si.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')
            if t_elem is not None and t_elem.text:
                shared_strings.append(t_elem.text)
            else:
                text_parts = [t.text for t in si.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if t.text]
                shared_strings.append("".join(text_parts))

    sheet_tree = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    rows = sheet_tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData/{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row')
    
    headers = []
    for c in rows[0].findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
        t = c.get('t')
        v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
        val = v.text if v is not None else ""
        if t == 's':
            val = shared_strings[int(val)]
        headers.append(val)

    data = []
    for r in rows[1:]:
        row_vals = {}
        for c in r.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
            r_coord = c.get('r') # e.g. A2, B2...
            # extract column letters
            col_letters = ''.join([ch for ch in r_coord if ch.isalpha()])
            # convert column letters to index (0-based)
            col_idx = 0
            for ch in col_letters:
                col_idx = col_idx * 26 + (ord(ch) - ord('A') + 1)
            col_idx -= 1
            
            t = c.get('t')
            v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
            val = v.text if v is not None else ""
            if t == 's':
                val = shared_strings[int(val)]
            if col_idx < len(headers):
                row_vals[headers[col_idx]] = val
        data.append(row_vals)

    print(f"Loaded {len(data)} rows.")
    for target in ['LAD', 'LCX', 'RCA', 'Cath']:
        counts = Counter([row.get(target, 'MISSING') for row in data])
        print(f"Target '{target}' distribution:", counts)

    # Also inspect any non-numeric or string values in a few columns
    print("\nSample values for some features:")
    for f in ['Sex', 'DM', 'HTN', 'Typical Chest Pain', 'VHD', 'Region RWMA', 'BBB', 'Function Class']:
        vals = Counter([row.get(f, 'MISSING') for row in data])
        print(f"Feature '{f}':", vals)
