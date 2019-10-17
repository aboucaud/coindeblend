#!/usr/bin/env python
# coding: utf-8
"""
Figure 8
--------
Bias and scatter of the predicted magnitude for the central galaxy

"""
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binned_statistic

from coindeblend.identity import paper_palette as PALETTE
from coindeblend.utilities import flux2mag

sns.set_palette(PALETTE)
sns.set_context("paper")
sns.set_style("ticks")

plt.rc("axes.spines", top=False)
plt.rc("font", size=20)
plt.rc("xtick", labelsize='large')
plt.rc("ytick", labelsize='large')

ZP_SEX = 25.67
ZP_NN = 25.96
MEAN_TICKS = [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]
STD_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] #, 1.2, 1.5]
MAG_TICKS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]


def get_stats(delta_mag, dmag, cut, std=False, bins=6):
    # stat = np.std if std else 'mean'
    stat = np.std if std else 'median'

    val, edge, _ = binned_statistic(delta_mag[cut], dmag[cut], statistic=stat, bins=bins)

    return val, edge


def main(datadir):
    data = pd.read_csv("output_catalog.csv")

    dis = data["distance"]

    g1_mag = data["g1_mag"]
    g2_mag = data["g2_mag"]

    delta_mag = g1_mag - g2_mag

    g1_sextractor_mag = flux2mag(data["g1_flux_sex"], ZP_SEX)
    g2_sextractor_mag = flux2mag(data["g2_flux_sex"], ZP_SEX)

    g1_b2f_mag = flux2mag(data["g1_flux_b2f"], ZP_NN)
    g2_b2f_mag = flux2mag(data["g2_flux_b2f"], ZP_NN)

    g1_b2m2f_mag = flux2mag(data["g1_flux_b2m2f"], ZP_NN)
    g2_b2m2f_mag = flux2mag(data["g2_flux_b2m2f"], ZP_NN)

    # SExtractor detection flag
    # -------------------------
    # 0 -> found only the central galaxy
    # 1 -> found only the companion galaxy
    # 2 -> found both galaxies
    # 3 -> found more than 2 objects
    central = data["detection"] == 0
    companion = data["detection"] == 1
    both = data["detection"] == 2
    multiple = data["detection"] == 3

    g1both = central | both
    g2both = companion | both


    bin_means_sextractor_co, _ = get_stats(delta_mag, g1_sextractor_mag - g1_mag, g1both)
    bin_std_sextractor_co, _ = get_stats(delta_mag, g1_sextractor_mag - g1_mag, g1both, std=True)
    bin_means_b2f_co, _ = get_stats(delta_mag, g1_b2f_mag - g1_mag, g1both)
    bin_std_b2f_co, _ = get_stats(delta_mag, g1_b2f_mag - g1_mag, g1both, std=True)
    bin_means_b2m2f_co, bin_edges = get_stats(delta_mag, g1_b2m2f_mag - g1_mag, g1both)
    bin_std_b2m2f_co, _ = get_stats(delta_mag, g1_b2m2f_mag - g1_mag, g1both, std=True)

    BINS_POSITION = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --------------------------------------------------------------
    # Figure 8
    # --------
    fig = plt.figure()
    fig.set_size_inches(9.75 * 1, 9.75 * 1)

    ax = fig.add_subplot(1, 1, 1)
    # ax = fig.add_subplot(2, 1, 1)
    ax.set_xlabel(r"${\rm mag}_{central} - {\rm mag}_{companion}$", fontsize=24, labelpad=20)
    ax.set_ylabel(r"Median ($\Delta_{mag})$", fontsize=24)
    ax.set_xlim(-2, 2)
    ax.set_xticks(MAG_TICKS)
    ax.set_xticklabels(MAG_TICKS)
    ax.set_ylim(-0.15, 0.15)
    ax.set_yticks(MEAN_TICKS)
    ax.plot(
        BINS_POSITION,
        bin_means_b2f_co,
        color=PALETTE[14],
        lw=5,
        label="  ",
    )
    ax.plot(
        BINS_POSITION,
        bin_means_b2m2f_co,
        color=PALETTE[9],
        lw=5,
        label="    ",
    )
    ax.plot(
        BINS_POSITION,
        bin_means_sextractor_co,
        color=PALETTE[4],
        lw=5,
        label=" ",
    )
    ax.plot([-2, 2], [0, 0], lw=2.5, ls="--", zorder=-1)
    ax.plot(
        BINS_POSITION,
        bin_means_sextractor_co,
        "d", ms=12, color=PALETTE[4],
    )
    ax.plot(
        BINS_POSITION,
        bin_means_b2f_co,
        "d", ms=12, color=PALETTE[14],
    )
    ax.plot(
        BINS_POSITION,
        bin_means_b2m2f_co,
        "d", ms=12, color=PALETTE[9],
    )

    ax2 = ax.twinx()
    ax2.set_ylabel(r"Scatter ($\Delta_{mag}$)", fontsize=24, labelpad=25)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_yticks(STD_TICKS)
    ax2.plot(
        BINS_POSITION,
        bin_std_b2f_co,
        color=PALETTE[14], lw=5, ls=":",
        label="  ",
    )
    ax2.plot(
        BINS_POSITION,
        bin_std_b2m2f_co,
        color=PALETTE[9], lw=5, ls=":",
        label="    ",
    )
    ax2.plot(
        BINS_POSITION,
        bin_std_sextractor_co,
        color=PALETTE[4], lw=5, ls=":",
        label=" ",
    )
    ax2.plot(
        BINS_POSITION,
        bin_std_b2f_co,
        "p", ms=12, color=PALETTE[14],
    )
    ax2.plot(
        BINS_POSITION,
        bin_std_b2m2f_co,
        "p", ms=12, color=PALETTE[9],
    )
    ax2.plot(
        BINS_POSITION,
        bin_std_sextractor_co,
        "p", ms=12, color=PALETTE[4],
    )

    l1 = ax2.legend(
        bbox_to_anchor=(1.12, 1.175),
        fontsize=24,
        frameon=False,
        ncol=3,
        columnspacing=1,
        handlelength=4.5,
    )

    ax.legend(
        bbox_to_anchor=(1.12, 1.25),
        fontsize=24,
        frameon=False,
        ncol=3,
        columnspacing=1,
        handlelength=4.5,
    )
    plt.text(-2.6, 1.17, "Median", fontsize=22)
    plt.text(-2.6, 1.09, "Scatter", fontsize=22)
    plt.text(-2.0, 1.25, "     Blend2Flux     Blend2Mask2Flux      Sextractor", fontsize=22)

    plt.subplots_adjust(left=0.175, right=0.85, top=0.78, bottom=0.13)
    plt.savefig("figures/figure8.png")


if __name__ == "__main__":
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')

    import sys
    try:
        datadir = sys.argv[1]
    except:
        sys.exit(
            "Please indicate the relative path to the data directory.\n"
            f"python {__name__} ../relative/path/to/data/"
        )

    main(datadir)
