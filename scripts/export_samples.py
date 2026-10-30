import io
import zipfile
import pandas as pd
from agents.data_normalizer import generate_multi_cloud_sample_fleet

fleet_df, _ = generate_multi_cloud_sample_fleet()
fleet_df.to_csv("data/samples/multi_cloud_fleet_sample.csv", index=False)
fleet_df.to_json("data/samples/multi_cloud_fleet_sample.json", orient="records", indent=2)

# Write XLSX
cols = list(fleet_df.columns)
shared_strings = []
string_to_id = {}

def get_str_id(s):
    s = str(s)
    if s not in string_to_id:
        string_to_id[s] = len(shared_strings)
        shared_strings.append(s)
    return string_to_id[s]

row_xmls = []

# Header row
hdr_cells = []
for c_idx, col in enumerate(cols):
    s_id = get_str_id(col)
    hdr_cells.append(f'<c t="s"><v>{s_id}</v></c>')
row_xmls.append(f'<row r="1">{"".join(hdr_cells)}</row>')

for r_idx, (_, row) in enumerate(fleet_df.iterrows(), 2):
    c_cells = []
    for col in cols:
        val = row[col]
        if pd.isna(val):
            c_cells.append('<c/>')
        elif isinstance(val, (int, float)):
            c_cells.append(f'<c><v>{val}</v></c>')
        else:
            s_id = get_str_id(str(val))
            c_cells.append(f'<c t="s"><v>{s_id}</v></c>')
    row_xmls.append(f'<row r="{r_idx}">{"".join(c_cells)}</row>')

sheet_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    f'<sheetData>{"".join(row_xmls)}</sheetData>'
    '</worksheet>'
)

sst_items = "".join([f'<si><t>{s}</t></si>' for s in shared_strings])
sst_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(shared_strings)}" uniqueCount="{len(shared_strings)}">'
    f'{sst_items}'
    '</sst>'
)

content_types_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
    '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
    '</Types>'
)

rels_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
    '</Relationships>'
)

wb_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    '<sheets><sheet name="CloudResources" sheetId="1" r:id="rId1"/></sheets>'
    '</workbook>'
)

wb_rels_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
    '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
    '</Relationships>'
)

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("[Content_Types].xml", content_types_xml)
    zf.writestr("_rels/.rels", rels_xml)
    zf.writestr("xl/workbook.xml", wb_xml)
    zf.writestr("xl/_rels/workbook.xml.rels", wb_rels_xml)
    zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    zf.writestr("xl/sharedStrings.xml", sst_xml)

with open("data/samples/multi_cloud_fleet_sample.xlsx", "wb") as f:
    f.write(buf.getvalue())

print("Successfully exported CSV, JSON, and XLSX samples.")

