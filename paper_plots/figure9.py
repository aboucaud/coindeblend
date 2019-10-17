#!/usr/bin/env python
# coding: utf-8
"""
Figure 9
--------
Bias and scatter as a function of morphological parameters

"""
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binned_statistic

from coindeblend.identity import paper_palette as PALETTE
from coindeblend.utilities import flux2mag

color_list = [
    PALETTE[i]
    for i in [3, 4, 7, 2, 8]
]

sns.set_palette(PALETTE)
sns.set_context("paper")
sns.set_style("ticks")

plt.rc("axes.spines", top=False, right=False)
plt.rc("font", size=20)
plt.rc("xtick", labelsize='large')
plt.rc("ytick", labelsize='large')

XMIN = -2
XMAX = 2
XTEXT = 0.83
YTEXT = 1.22
MEAN_TICKS = [-0.05, 0.0, 0.05, 0.1, 0.15]
STD_TICKS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
MAG_TICKS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

ZP_SEX = 25.67
ZP_NN = 25.96

TYPE_LIST = [
    "sph",
    "sphd",
    "disk",
    "irr",
    None,
]


def get_stats(delta_mag, dmag, gtype=None, std=False, bins=6):
    stat = np.std if std else 'median'

    if gtype is None:
        val, edge, _ = binned_statistic(delta_mag, dmag, statistic=stat, bins=bins)
    else:
        cut = g1_morp h  == gtype
        val, edge, _  =  binned_statistic(delta_mag[cut], dmag[cut], statistic=stat, bins=bins)

    return val, edge


def plot_ax(ax, title, delta_mag, dmag, ymin=None, ymax=None,
            std=False, ylabel=False):
    ax.grid(True)
    ax.set_title(title, fontsize=26, fontstyle='italic')

    marker = 'p' if std else 'd'
    lstyle = ':' if std else '-'
    label = r'Scatter($\Delta_{mag}$)' if std else r'Median($\Delta_{mag})$'

    ax.set_xlabel(r'${\rm mag}_{central} - {\rm mag}_{companion}$', fontsize=24, labelpad=20)
    if ylabel:
        ax.set_ylabel(label, fontsize=24)

    for gtype, color in zip(TYPE_LIST, color_list):
        val, edges = get_stats(delta_mag, dmag, gtype, std=std)
        mag_bins = (edges[:-1] + edges[1:]) / 2
        ax.plot(mag_bins, val, color=color, lw=5, ls=lstyle, label=' ')
        ax.plot(mag_bins, val, marker, ms=12, color=color)

    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks(MAG_TICKS)
    ax.set_xticklabels(MAG_TICKS)
    ax.set_ylim(ymin, ymax)


def main():
    data = pd.read_csv(f"{datadir}/output_catalog.csv")

    dis = data["distance"]

    g1_morph = data["g1_type"]
    g2_morph = data["g2_type"]

    g1_mag = data["g1_mag"]
    g2_mag = data["g2_mag"]

    g1_sextractor_mag = flux2mag(data["g1_flux_sex"], ZP_SEX)
    g2_sextractor_mag = flux2mag(data["g2_flux_sex"], ZP_SEX)

    g1_b2f_mag = flux2mag(data["g1_flux_b2f"], ZP_NN)
    g2_b2f_mag = flux2mag(data["g2_flux_b2f"], ZP_NN)

    g1_b2m2f_mag = flux2mag(data["g1_flux_b2m2f"], ZP_NN)
    g2_b2m2f_mag = flux2mag(data["g2_flux_b2m2f"], ZP_NN)

    delta_mag = g1_mag - g2_mag
    dmag_b2f = g1_b2f_mag - g1_mag
    dmag_b2m2f = g1_b2m2f_mag - g1_mag
    dmag_sex = g1_sex_mag - g1_mag

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

    dmag_sex = dmag_sex[g1both]
    delta_mag_sex = delta_mag[g1both]

    # ---------------------------------------------------------------
    # Figure 9
    # --------
    fig = plt.figure()
    fig.set_size_inches(9.75*3, 9.75*2)

    ax1 = plt.subplot(2,3,1)
    plot_ax(ax1, 'blend2flux', delta_mag, dmag_b2f, ymin=-0.07, ymax=0.07, ylabel=True)
    ax2 = plt.subplot(2, 3, 2)
    plot_ax(ax2, 'blend2mask2flux', delta_mag, dmag_b2m2f, ymin=-0.07, ymax=0.07, ylabel=True)
    ax3 = plt.subplot(2, 3, 3)
    plot_ax(ax3, 'SExtractor', delta_mag_sex, dmag_sex, ymin=-0.4, ymax=0.4, ylabel=True)
    ax4 = plt.subplot(2, 3, 4)
    plot_ax(ax4, 'blend2flux', delta_mag, dmag_b2f, ymin=0.0, std=True, ylabel=True)
    ax5 = plt.subplot(2, 3, 5)
    plot_ax(ax5, 'blend2mask2flux', delta_mag, dmag_b2m2f, ymin=0.0, std=True, ylabel=True)
    ax6 = plt.subplot(2, 3, 6)
    plot_ax(ax6, 'SExtractor', delta_mag_sex, dmag_sex, ymin=0.0, std=True, ylabel=True)

    ax1.legend(bbox_to_anchor=(2.65, 1.3), fontsize=24, frameon=False, ncol=5, columnspacing=1.75, handlelength=4.5)
    ax4.legend(bbox_to_anchor=(2.65, 2.5), fontsize=24, frameon=False, ncol=5, columnspacing=1.75, handlelength=4.5)

    plt.text(XTEXT - 0.03, YTEXT, 'Median', transform=ax1.transAxes, fontsize=22)
    plt.text(XTEXT - 0.03, YTEXT - 0.1, 'Scatter', transform=ax1.transAxes, fontsize=22)
    plt.text(XTEXT + 0.2, YTEXT + 0.08, 'Spheroids           Bulge + Disk             Disks                  Irregular                     All', transform=ax1.transAxes, fontsize=22)

    plt.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.08, wspace=0.3, hspace=0.3)
    plt.savefig("figures/figure9.png")


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
