import zipfile
import xml.etree.ElementTree as ET

xlsx_path = "data/extention of Z-Alizadeh sani dataset.xlsx"

with zipfile.ZipFile(xlsx_path, 'r') as z:
    print("Files in xlsx:", [n for n in z.namelist() if n.endswith('.xml')])
    # Read shared strings
    shared_strings = []
    if 'xl/sharedStrings.xml' in z.namelist():
        tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
        for si in tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
            # Text can be in t or r/t
            t_elem = si.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')
            if t_elem is not None and t_elem.text:
                shared_strings.append(t_elem.text)
            else:
                text_parts = [t.text for t in si.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if t.text]
                shared_strings.append("".join(text_parts))
    print(f"Total shared strings: {len(shared_strings)}")
    
    # Read sheet1
    sheet_tree = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    rows = sheet_tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData/{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row')
    print(f"Total rows in sheet1: {len(rows)}")
    
    # Print header row
    first_row = rows[0]
    headers = []
    for c in first_row.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
        t = c.get('t')
        v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
        val = v.text if v is not None else ""
        if t == 's':
            val = shared_strings[int(val)]
        headers.append(val)
    print("Headers count:", len(headers))
    print("Headers:", headers)
