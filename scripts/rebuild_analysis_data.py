#!/usr/bin/env python3
"""Rebuild analysis_data.json using the latest baselines and KaRMA scores."""
import argparse
import json
import yaml
from pathlib import Path
from scipy import stats
import numpy as np

# No options; parsing still gives a working --help and rejects stray arguments
# before anything is overwritten.
argparse.ArgumentParser(description=__doc__).parse_args()

ROOT = Path(__file__).resolve().parent.parent

# Load KaRMA scores
with open(ROOT / "results" / "16_hand_batch" / "summary.yaml") as f:
    karma_data = yaml.safe_load(f)

# Load the most recent baselines batch. batch_<timestamp> dirs sort chronologically,
# so regenerating with `python -m baselines.run_baselines` (which writes a fresh
# batch) is picked up automatically here.
_batch_dirs = sorted((ROOT / "results" / "baselines").glob("batch_*"))
if not _batch_dirs:
    raise FileNotFoundError(
        "No results/baselines/batch_* found. Run: python -m baselines.run_baselines"
    )
baseline_batch = _batch_dirs[-1]
with open(baseline_batch / "summary.yaml") as f:
    baselines_data = yaml.safe_load(f)

# Build name mapping for baselines (robot_name -> entry)
# Normalize names to match between datasets
NAME_MAP = {
    "ARMS": "ARMS_skel",
    "DLR_handmodel": "DLR",
    "ability_hand_right": "ability_hand_right",
    "allegro_hand_right": "allegro",
    "dclaw_gripper": "dclaw",
    "Unitree_Dex3_Right": "dex3",
    "dex5_right": "dex5",
    "inspire_hand_right": "inspire",
    "leap_hand": "leap",
    "orcahand_right": "orcahand",
    "shadowhand_motor": "shadowhand",
    "sharpa": "sharpa",
    "schunk_svh_hand_right": "svh",
    "wuji_hand_right": "wuji",
    "xhand1_right": "xhand1",
    "xhandlite_right": "xhandlite",
}

baselines_by_name = {}
for entry in baselines_data["results"]:
    short = NAME_MAP.get(entry["robot_name"], entry["robot_name"])
    baselines_by_name[short] = entry

# Build combined hand records
hands = []
for entry in karma_data["rows"]:
    if "error" in entry:
        continue
    short = NAME_MAP.get(entry["robot"], entry["robot"])
    bl = baselines_by_name.get(short)
    if bl is None:
        print(f"WARNING: no baselines for {short}")
        continue

    # total_range from baselines: mean_range_rad * n_joints_total
    total_range = round(bl["mean_range_rad"] * bl["n_joints_total"], 2)

    hands.append({
        "hand": short,
        "karma_t": entry["KaRMA_T"],
        "karma_r": entry["KaRMA_R"],
        "voxels": entry["n_voxels"],
        "l_ref_m": entry["L_ref_mm"] / 1000.0,
        "opposability": round(bl["opposability_index"], 4),
        "total_dof": bl["n_joints_total"],
        "total_range_rad": total_range,
        "yoshikawa": bl["combined_yoshikawa_mean"],
        "gci": bl["combined_gci"],
    })

# Sort by karma_t descending for ranking
hands.sort(key=lambda h: h["karma_t"], reverse=True)

# Assign KaRMA-T ranks
for i, h in enumerate(hands):
    h["karma_t_rank"] = i + 1

# Assign KaRMA-R ranks
r_sorted = sorted(hands, key=lambda h: h["karma_r"], reverse=True)
for i, h in enumerate(r_sorted):
    h["karma_r_rank"] = i + 1

# Assign opposability ranks
opp_sorted = sorted(hands, key=lambda h: h["opposability"], reverse=True)
for i, h in enumerate(opp_sorted):
    h["opp_rank"] = i + 1

# Compute rank residuals
for h in hands:
    h["rank_residual"] = h["opp_rank"] - h["karma_t_rank"]
    h["rank_residual_r"] = h["opp_rank"] - h["karma_r_rank"]

# Spearman correlations
karma_t_vals = [h["karma_t"] for h in hands]
karma_r_vals = [h["karma_r"] for h in hands]
opp_vals = [h["opposability"] for h in hands]
dof_vals = [h["total_dof"] for h in hands]
range_vals = [h["total_range_rad"] for h in hands]

rho_opp_t, p_opp_t = stats.spearmanr(opp_vals, karma_t_vals)
rho_opp_r, p_opp_r = stats.spearmanr(opp_vals, karma_r_vals)
rho_dof_t, p_dof_t = stats.spearmanr(dof_vals, karma_t_vals)
rho_range_t, p_range_t = stats.spearmanr(range_vals, karma_t_vals)
rho_yosh_t, p_yosh_t = stats.spearmanr([h["yoshikawa"] for h in hands], karma_t_vals)
rho_gci_t, p_gci_t = stats.spearmanr([h["gci"] for h in hands], karma_t_vals)
rho_tr, p_tr = stats.spearmanr(karma_t_vals, karma_r_vals)

# Delta = (T rank - R rank) vs each baseline: supports Section VI-B's claim that
# "no baseline predicts the T-R divergence" (all weak; |rho_s| <= 0.31, p > 0.24).
delta_vals = [h["karma_t_rank"] - h["karma_r_rank"] for h in hands]
yosh_vals = [h["yoshikawa"] for h in hands]
gci_vals = [h["gci"] for h in hands]
delta_baseline = {
    "opposability": stats.spearmanr(delta_vals, opp_vals),
    "total_dof": stats.spearmanr(delta_vals, dof_vals),
    "total_range": stats.spearmanr(delta_vals, range_vals),
    "yoshikawa": stats.spearmanr(delta_vals, yosh_vals),
    "gci": stats.spearmanr(delta_vals, gci_vals),
}
delta_max_abs_rho = max(abs(r) for r, _ in delta_baseline.values())
delta_min_p = min(p for _, p in delta_baseline.values())

# Rank displacement analysis for T
abs_displacements_t = [abs(h["rank_residual"]) for h in hands]
misranked_ge2_t = sum(1 for d in abs_displacements_t if d >= 2)
any_change_t = sum(1 for d in abs_displacements_t if d > 0)

# Pairwise inversions for T
n_hands = len(hands)
inversions_t = 0
total_pairs = 0
for i in range(n_hands):
    for j in range(i + 1, n_hands):
        total_pairs += 1
        t_order = hands[i]["karma_t_rank"] < hands[j]["karma_t_rank"]
        o_order = hands[i]["opp_rank"] < hands[j]["opp_rank"]
        if t_order != o_order:
            inversions_t += 1

# Rank displacement analysis for R
abs_displacements_r = [abs(h["rank_residual_r"]) for h in hands]
misranked_ge2_r = sum(1 for d in abs_displacements_r if d >= 2)
any_change_r = sum(1 for d in abs_displacements_r if d > 0)

inversions_r = 0
for i in range(n_hands):
    for j in range(i + 1, n_hands):
        r_order = hands[i]["karma_r_rank"] < hands[j]["karma_r_rank"]
        o_order = hands[i]["opp_rank"] < hands[j]["opp_rank"]
        if r_order != o_order:
            inversions_r += 1

# Outliers (|rank_residual| >= 3)
outliers_t = [h for h in hands if abs(h["rank_residual"]) >= 3]
outliers_r = [h for h in hands if abs(h["rank_residual_r"]) >= 3]

result = {
    "description": f"KaRMA vs workspace baselines analysis (rebuilt with {baseline_batch.name} baselines)",
    "karma_source": "results/16_hand_batch",
    "baseline_source": f"results/baselines/{baseline_batch.name} (Sobol, convex hull)",
    "correlations": {
        "opposability_vs_karma_t": {
            "rho": round(rho_opp_t, 4),
            "p": round(p_opp_t, 6),
        },
        "opposability_vs_karma_r": {
            "rho": round(rho_opp_r, 4),
            "p": round(p_opp_r, 6),
        },
        "total_dof_vs_karma_t": {
            "rho": round(rho_dof_t, 4),
            "p": round(p_dof_t, 6),
        },
        "total_range_vs_karma_t": {
            "rho": round(rho_range_t, 4),
            "p": round(p_range_t, 6),
        },
        "yoshikawa_vs_karma_t": {
            "rho": round(rho_yosh_t, 4),
            "p": round(p_yosh_t, 6),
        },
        "gci_vs_karma_t": {
            "rho": round(rho_gci_t, 4),
            "p": round(p_gci_t, 6),
        },
        "karma_t_vs_karma_r": {
            "rho": round(rho_tr, 4),
            "p": round(p_tr, 6),
        },
    },
    "delta_vs_baseline_correlations": {
        k: {"rho": round(r, 4), "p": round(p, 4)}
        for k, (r, p) in delta_baseline.items()
    },
    "delta_vs_baseline_summary": {
        "max_abs_rho": round(delta_max_abs_rho, 4),
        "min_p": round(delta_min_p, 4),
        "paper_claim": "all weak: |rho_s| <= 0.31, p > 0.24",
    },
    "hands": hands,
    "rank_displacement": {
        "karma_t": {
            "mean_abs_displacement": round(np.mean(abs_displacements_t), 2),
            "fraction_misranked_ge2": f"{misranked_ge2_t}/{n_hands} ({round(100*misranked_ge2_t/n_hands)}%)",
            "fraction_any_change": f"{any_change_t}/{n_hands} ({round(100*any_change_t/n_hands)}%)",
            "pairwise_inversions": f"{inversions_t}/{total_pairs} ({round(100*inversions_t/total_pairs)}%)",
            "n_outliers_ge3": len(outliers_t),
        },
        "karma_r": {
            "mean_abs_displacement": round(np.mean(abs_displacements_r), 2),
            "fraction_misranked_ge2": f"{misranked_ge2_r}/{n_hands} ({round(100*misranked_ge2_r/n_hands)}%)",
            "fraction_any_change": f"{any_change_r}/{n_hands} ({round(100*any_change_r/n_hands)}%)",
            "pairwise_inversions": f"{inversions_r}/{total_pairs} ({round(100*inversions_r/total_pairs)}%)",
            "n_outliers_ge3": len(outliers_r),
        },
    },
    "outliers": outliers_t,
    "outliers_r": outliers_r,
}

out_path = ROOT / "results" / "karma_vs_workspace" / "analysis_data.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"Wrote {out_path}")
print("\nCorrelations:")
print(f"  opposability vs KaRMA-T: rho={rho_opp_t:.4f}, p={p_opp_t:.6f}")
print(f"  opposability vs KaRMA-R: rho={rho_opp_r:.4f}, p={p_opp_r:.6f}")
print(f"  total_dof vs KaRMA-T:    rho={rho_dof_t:.4f}, p={p_dof_t:.6f}")
print(f"  total_range vs KaRMA-T:  rho={rho_range_t:.4f}, p={p_range_t:.6f}")
print(f"  Yoshikawa vs KaRMA-T:    rho={rho_yosh_t:.4f}, p={p_yosh_t:.6f}")
print(f"  GCI vs KaRMA-T:          rho={rho_gci_t:.4f}, p={p_gci_t:.6f}")
print(f"  KaRMA-T vs KaRMA-R:      rho={rho_tr:.4f}, p={p_tr:.6f}")
print("\nDelta (T rank - R rank) vs baselines [Section VI-B]:")
for _k, (_r, _p) in delta_baseline.items():
    print(f"  {_k:>13s}: rho={_r:+.4f}, p={_p:.4f}")
print(f"  -> max|rho|={delta_max_abs_rho:.4f}, min p={delta_min_p:.4f} (paper: |rho|<=0.31, p>0.24)")
print("\nRank displacement (T):")
print(f"  misranked >=2: {misranked_ge2_t}/{n_hands} ({100*misranked_ge2_t/n_hands:.0f}%)")
print(f"  pairwise inversions: {inversions_t}/{total_pairs} ({100*inversions_t/total_pairs:.0f}%)")
print(f"  outliers (>=3): {len(outliers_t)}")
for o in outliers_t:
    print(f"    {o['hand']}: opp_rank={o['opp_rank']} -> karma_t_rank={o['karma_t_rank']} (residual={o['rank_residual']})")
print("\nRank displacement (R):")
print(f"  misranked >=2: {misranked_ge2_r}/{n_hands} ({100*misranked_ge2_r/n_hands:.0f}%)")
print(f"  pairwise inversions: {inversions_r}/{total_pairs} ({100*inversions_r/total_pairs:.0f}%)")
print(f"  outliers (>=3): {len(outliers_r)}")
for o in outliers_r:
    print(f"    {o['hand']}: opp_rank={o['opp_rank']} -> karma_r_rank={o['karma_r_rank']} (residual={o['rank_residual_r']})")

# Print full ranking table for verification
print(f"\n{'Hand':>15s}  T_rk  R_rk  Opp_rk  resid_T  resid_R  opp_val")
for h in hands:
    print(f"{h['hand']:>15s}  {h['karma_t_rank']:>4d}  {h['karma_r_rank']:>4d}  {h['opp_rank']:>6d}  {h['rank_residual']:>+5d}    {h['rank_residual_r']:>+5d}    {h['opposability']:.4f}")
