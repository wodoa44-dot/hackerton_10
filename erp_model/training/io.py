"""Training CSV and JSON I/O without importing estimator implementations."""
import csv
import json
from pathlib import Path

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    if not rows or not {'품목명','규격명','품목코드'}<=rows[0].keys():
        raise ValueError('Missing CSV columns')
    return rows

def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
