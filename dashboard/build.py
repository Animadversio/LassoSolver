"""Convert the dashboard's editable TSV tables to the JSON files that get uploaded."""
import csv, json, pathlib, re
here = pathlib.Path(__file__).parent
NUM = {"targets", "time_s", "test_r2", "test_r2_rel", "nnz"}
for name in ("ideas", "results"):
    rows = []
    for r in csv.DictReader(open(here / f"{name}.tsv"), delimiter="\t"):
        row = {k: (None if v == "" else float(v)) if k in NUM else v for k, v in r.items()}
        if name == "results":  # stable row key for the dashboard's table marks
            key = "-".join(str(r[k]) for k in ("tier", "solver", "hardware", "targets", "note"))
            row = {"id": re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-"), **row}
        rows.append(row)
    json.dump(rows, open(here / f"{name}.json", "w"), indent=1)
    print(name, len(rows))

page = (here / "page.html").read_text()
json.dump({"text": page}, open(here / "page.doc.json", "w"))
