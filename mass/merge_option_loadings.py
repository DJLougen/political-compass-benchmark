#!/usr/bin/env python3
"""
Merge per-judge option-loading annotations into one frozen CROSS-FAMILY consensus CSV.

Each input is the output of annotate_option_loadings.py for one judge family (e.g. Spark/Qwen
and CLI Fable/Claude). We key on (item_id, option_index) and, per dimension, snap the mean of
the judges' signed loadings to {-1,-0.5,0,+0.5,+1}. is_personal is a majority vote. We record
`judge_agreement` (cross-family agreement on the dims any judge marked) so the reference is
usable as a frozen set with a trust signal, independent of any single model family.

  python mass/merge_option_loadings.py \
      --inputs data/source_option_loadings_spark.csv,data/source_option_loadings_fable.csv \
      --names spark-latest,claude-fable-5 \
      --out data/source_option_loadings.csv
"""
import argparse
import csv
import os

DIMS = ["economic", "social", "civil_liberties", "authority", "institutions",
        "national_identity", "foreign_policy", "environment", "religion",
        "technocracy", "populism", "personal"]
ALLOWED = [-1.0, -0.5, 0.0, 0.5, 1.0]
csv.field_size_limit(10 ** 7)


def snap(x):
    return min(ALLOWED, key=lambda a: abs(a - x))


def load(path):
    """-> {(item_id, opt_idx): row}, keeping order of first file."""
    out, order = {}, []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        k = (r["item_id"], int(r["option_index"]))
        out[k] = r
        order.append(k)
    return out, order


def usable(r):
    return r is not None and r.get("primary_dimension") != "ERROR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="comma-list of per-judge CSVs")
    ap.add_argument("--names", default="", help="comma-list of judge labels (default: file stem)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = args.inputs.split(",")
    names = args.names.split(",") if args.names else [os.path.basename(p) for p in paths]

    tables, order = [], None
    for p in paths:
        t, o = load(p)
        tables.append(t)
        if order is None:
            order = o
    # meta comes from the first usable judge for each key
    header = (["item_id", "source", "source_key", "option_index", "option_text",
               "n_options", "options_ordered", "is_personal", "primary_dimension",
               "primary_load"] + [f"load_{d}" for d in DIMS] +
              ["judge_agreement", "within_judge_conf", "n_judges", "judges"])
    written = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in order:
            rows = [t.get(k) for t in tables]
            meta = next((r for r in rows if usable(r)), None) or next((r for r in rows if r), None)
            if meta is None:
                continue
            present = [(nm, r) for nm, r in zip(names, rows) if usable(r)]
            nj = len(present)
            personal_votes = sum(1 for _, r in present if r.get("is_personal") == "True")
            is_personal = nj > 0 and personal_votes > nj / 2.0
            within = [float(r.get("confidence", 0) or 0) for _, r in present]
            within_conf = round(min(within), 3) if within else 0.0

            loadvec, agree_terms = [], []
            if is_personal:
                loadvec = [0.0] * len(DIMS)
                loadvec[DIMS.index("personal")] = 1.0
            else:
                for d in DIMS:
                    if d == "personal":
                        loadvec.append(0.0)
                        continue
                    vals = [float(r.get(f"load_{d}", 0) or 0) for _, r in present]
                    sv = snap(sum(vals) / nj) if nj else 0.0
                    loadvec.append(sv)
                    if any(abs(v) > 1e-9 for v in vals):   # a dim some judge tagged
                        agree_terms.append(sum(1 for v in vals if snap(v) == sv) / nj)
            agreement = round(sum(agree_terms) / len(agree_terms), 3) if agree_terms else 1.0

            nonzero = {d: v for d, v in zip(DIMS, loadvec) if abs(v) > 1e-9 and d != "personal"}
            if is_personal:
                pdim, pload = "personal", 1.0
            elif nonzero:
                pdim = max(nonzero, key=lambda d: abs(nonzero[d]))
                pload = nonzero[pdim]
            else:
                pdim, pload = "none", 0.0
            w.writerow([meta["item_id"], meta["source"], meta["source_key"], k[1],
                        meta["option_text"], meta["n_options"], meta["options_ordered"],
                        is_personal, pdim, pload] + loadvec +
                       [agreement, within_conf, nj, ",".join(nm for nm, _ in present)])
            written += 1
    print(f"Merged {len(paths)} judges -> {args.out}  ({written} option rows)")


if __name__ == "__main__":
    main()
