#!/usr/bin/env python3
"""
Aggregate 4 image-stitching CSV logs.

*   Renames `image_idx` ➜ `problem_id`  (e.g. “malibu_1”)
*   Derives three metrics relative to each problem’s baseline (CG+none):
      – speed-up           (elapsed_ms)
      – iteration_reduction(%)
      – mem_saving(%)      (mem_meas_kb)
*   Computes the geometric mean of those metrics **per configuration**
    across all problems.
*   Writes `gmean_per_config.csv`
"""

import sys, pathlib, pandas as pd, numpy as np
from scipy.stats import gmean

# ----------------------------------------------------------------------
# 1 – file list (override with argv)
files = {
    "field" : "log_field.csv",
    "sh"    : "log_sh.csv",
    "malibu": "log_malibu.csv",
    "gj"    : "log_gj.csv"
}
if len(sys.argv) == 5:                   # allow CLI override
    files = dict(zip(files.keys(), sys.argv[1:]))

# ----------------------------------------------------------------------
# 2 – load and merge
dfs = []
for name, path in files.items():
    df = pd.read_csv(path)
    df["problem_id"] = (name + "_" + df["image_idx"].astype(str))
    dfs.append(df)
all_df = pd.concat(dfs, ignore_index=True)

# ----------------------------------------------------------------------
# 3 – identify baseline rows  (solver=cg, precond=none)
base = (all_df["solver"].str.lower() == "cg") & \
       (all_df["precond"].str.lower() == "none")

# sanity
if base.sum() == 0:
    raise RuntimeError("No baseline rows (cg+none) found!")

# ----------------------------------------------------------------------
# 4 – compute metrics per problem_id relative to its baseline
def rel_metric(group: pd.DataFrame):
    """Return group with 3 new cols relative to that group's baseline row."""
    b = group[base.loc[group.index]].iloc[0]  # baseline row
    for idx, row in group.iterrows():
        speed   = b["elapsed_ms"]/row["elapsed_ms"]
        iterred = (b["iterations"] - row["iterations"]) / b["iterations"] * 100
        memsav  = (b["mem_meas_kb"] - row["mem_meas_kb"]) / b["mem_meas_kb"] * 100
        all_df.loc[idx, "speedup"]               = speed
        all_df.loc[idx, "iteration_reduction_%"] = iterred
        all_df.loc[idx, "mem_saving_%"]          = memsav
    return group

_ = all_df.groupby("problem_id", group_keys=False).apply(rel_metric)

# drop baseline rows themselves (speedup == 1, etc. – not meaningful)
all_df = all_df[~base]

# ----------------------------------------------------------------------
# 5 – encode “configuration” string
all_df["configuration"] = (all_df["solver"] + "+" +
                           all_df["precond"] + "+" +
                           all_df["ordering"].fillna(""))

# ----------------------------------------------------------------------
# 6 – geometric mean (and arithmetic mean for robustness)
def agg(df):
    d = {}
    for col in ["speedup",
                "iteration_reduction_%", "mem_saving_%"]:
        vals = df[col].dropna().values
        d["gmean_"+col] = gmean(np.maximum(vals, 1e-12))  # avoid zero
        d["mean_"+col]  = vals.mean()
    return pd.Series(d)

gmean_df = all_df.groupby("configuration").apply(agg).reset_index()

# ----------------------------------------------------------------------
# 7 – save
gmean_df.to_csv("gmean_per_config.csv", index=False)
print("Wrote gmean_per_config.csv")

