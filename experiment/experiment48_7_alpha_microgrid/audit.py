#!/usr/bin/env python3
import csv
from pathlib import Path
rows=list(csv.DictReader(Path(__file__).parent.joinpath("results/fold_metrics.csv").open()))
assert len(rows)==240 and all(r["prediction_sha_match"]=="True" for r in rows)
print("AUDIT_PASS",len(rows))
