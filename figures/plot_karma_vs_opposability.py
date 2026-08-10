#!/usr/bin/env python3
"""Scatter plot: KaRMA-T and KaRMA-R vs workspace opposability.

The standout outlier (Allegro, displaced >= 3 rank positions in both scores)
is diamond-marked. Zero-intersection hands are shown in a separate strip at x=0.
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# No options; parsing still gives a working --help and rejects stray arguments
# before any figure files are written.
argparse.ArgumentParser(description=__doc__).parse_args()

# Embed TrueType (Type 42) fonts rather than matplotlib's default Type 3, so the
# resulting PDF passes IEEE PDF eXpress font checks.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

FIG_DIR = Path(__file__).resolve().parent
ROOT = FIG_DIR.parent
DATA = ROOT / "results" / "karma_vs_workspace" / "analysis_data.json"
OUT_DIR = FIG_DIR

with open(DATA) as f:
    data = json.load(f)

hands = data["hands"]

# Separate zero vs nonzero opposability
zero_hands = [h for h in hands if h["opposability"] == 0]
nonzero_hands = [h for h in hands if h["opposability"] > 0]

fig, ax = plt.subplots(figsize=(4.5, 3.5))

# --- Nonzero opposability (main scatter region) ---
opp_nz = np.array([h["opposability"] for h in nonzero_hands])
t_nz = np.array([h["karma_t"] for h in nonzero_hands])
r_nz = np.array([h["karma_r"] for h in nonzero_hands])

ax.scatter(opp_nz, t_nz, s=30, c="#2166ac", marker="o", zorder=5, label="KaRMA-T")
ax.scatter(opp_nz, r_nz, s=30, c="#d95f02", marker="s", zorder=5, label="KaRMA-R")

# Diamond the standout outlier: only Allegro is displaced >=3 rank positions in BOTH
# KaRMA-T and KaRMA-R (D'Claw is >=3 only in R; the lower near-tied hands are noise).
outlier_names = {"allegro"}
for h in nonzero_hands:
    if h["hand"] in outlier_names:
        x = h["opposability"]
        ax.scatter(x, h["karma_t"], s=70, facecolors="none", edgecolors="#2166ac",
                   marker="D", linewidths=1.2, zorder=6)
        ax.scatter(x, h["karma_r"], s=70, facecolors="none", edgecolors="#d95f02",
                   marker="D", linewidths=1.2, zorder=6)

# --- Zero opposability strip ---
x_zero = 2e-5
band_lo, band_hi = 1.2e-5, 3.3e-5

ax.axvspan(band_lo, band_hi, color="#f0f0f0", zorder=0)
ax.text(x_zero, 0.25, "Zero\nInt.", ha="center", va="bottom",
        fontsize=7, color="#666666", style="italic")

opp_z_x = np.full(len(zero_hands), x_zero)
t_z = np.array([h["karma_t"] for h in zero_hands])
r_z = np.array([h["karma_r"] for h in zero_hands])

ax.scatter(opp_z_x, t_z, s=30, c="#2166ac", marker="o", zorder=5)
ax.scatter(opp_z_x, r_z, s=30, c="#d95f02", marker="s", zorder=5)

# Mark zero-intersection outliers
for h in zero_hands:
    if h["hand"] in outlier_names:
        ax.scatter(x_zero, h["karma_t"], s=70, facecolors="none", edgecolors="#2166ac",
                   marker="D", linewidths=1.2, zorder=6)
        ax.scatter(x_zero, h["karma_r"], s=70, facecolors="none", edgecolors="#d95f02",
                   marker="D", linewidths=1.2, zorder=6)

# --- Trend lines (nonzero only) ---
log_opp = np.log10(opp_nz)
log_t = np.log10(t_nz)
log_r = np.log10(r_nz)

fit_t = np.polyfit(log_opp, log_t, 1)
fit_r = np.polyfit(log_opp, log_r, 1)

x_fit = np.linspace(np.log10(opp_nz.min()) - 0.15, np.log10(opp_nz.max()) + 0.15, 50)
ax.plot(10**x_fit, 10**np.polyval(fit_t, x_fit), "--", color="#2166ac", alpha=0.35, lw=1.0, zorder=2)
ax.plot(10**x_fit, 10**np.polyval(fit_r, x_fit), "--", color="#d95f02", alpha=0.35, lw=1.0, zorder=2)

# --- Stats box ---
rho_t = data["correlations"]["opposability_vs_karma_t"]["rho"]
rho_r = data["correlations"]["opposability_vs_karma_r"]["rho"]
disp_t = data["rank_displacement"]["karma_t"]
disp_r = data["rank_displacement"]["karma_r"]

stats_text = (
    f"T: $\\rho_s$={rho_t:.2f}, {disp_t['fraction_misranked_ge2']} misranked\n"
    f"R: $\\rho_s$={rho_r:.2f}, "
    f"{disp_r['fraction_misranked_ge2']} misranked"
)
ax.text(0.45, 0.03, stats_text, transform=ax.transAxes, fontsize=7.5,
        va="bottom", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.95))

# --- Formatting ---
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(8e-6, 0.5)
ax.set_ylim(1.5e-4, 0.6)
ax.set_xlabel("Workspace Opposability (intersection / $L_{ref}^3$)", fontsize=11)
ax.set_ylabel("KaRMA Score", fontsize=11)
ax.tick_params(labelsize=9)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9,
          handletextpad=0.4, borderpad=0.3,
          handlelength=1.0)
ax.set_title("KaRMA vs Workspace Opposability", fontsize=12, pad=8)
ax.grid(True, which="major", ls="-", alpha=0.15)
ax.grid(True, which="minor", ls=":", alpha=0.08)

plt.tight_layout()
plt.savefig(str(OUT_DIR / "karma_vs_opposability.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(str(OUT_DIR / "karma_vs_opposability.png"), dpi=200, bbox_inches="tight")
print(f"Saved to {OUT_DIR / 'karma_vs_opposability.pdf'}")
print(f"Saved to {OUT_DIR / 'karma_vs_opposability.png'}")
