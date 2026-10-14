import zipfile
import xml.etree.ElementTree as ET
import csv

xlsx_path = "data/extention of Z-Alizadeh sani dataset.xlsx"
csv_path = "data/z_alizadeh_sani_extended.csv"

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
        headers.append(val.strip())

    all_rows = []
    for r in rows[1:]:
        row_cells = {i: "" for i in range(len(headers))}
        for c in r.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
            r_coord = c.get('r')
            col_letters = ''.join([ch for ch in r_coord if ch.isalpha()])
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
                row_cells[col_idx] = val.strip()
        all_rows.append([row_cells[i] for i in range(len(headers))])

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(all_rows)

print(f"Successfully converted to {csv_path} with {len(all_rows)} rows and {len(headers)} columns.")
