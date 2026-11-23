# Dashboard

Live page: https://claude.ai/artifact/PuVGTXkGsJ4iSpr7iecEg8 (a "Data" artifact; private unless shared).

- `ideas.tsv`: one row per solver idea. `status` is one of `running`, `planned`, `later`, `kept`, `dropped`;
  `verdict` is the one-line conclusion with numbers. Edited by the coordinator session.
- `results.tsv`: one row per timed run. Seeded with the Dec 2024 numbers; later rounds are generated from
  `tables/results.jsonl` (the harness output).
- `page.html`: the dashboard page (stored in the artifact as `files/index.html`).

To update: edit the TSVs, run `python dashboard/build.py`, upload `ideas.json` / `results.json` to the artifact
as assets, then point `datasets/ideas` / `datasets/results` at the new `/_blob/<id>` urls and bump `updated.at`.
For a page change, also `set` `files/index.html` from `page.doc.json`.
