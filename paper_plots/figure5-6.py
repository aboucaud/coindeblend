#!/usr/bin/env python
# coding: utf-8
"""
Figure 5-6
----------
Predicted vs. True magnitude with statistics and histograms embedded
for the three different models

"""
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from coindeblend.stats import mean_absolute_percentage_error
from coindeblend.identity import paper_palette as PALETTE
from coindeblend.utilities import flux2mag


sns.set_palette(PALETTE)
sns.set_context("paper")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)
plt.rc("font", size=20)
plt.rc("xtick", labelsize='large')
plt.rc("ytick", labelsize='large')

ZP_SEX = 25.67
ZP_NN = 25.96
XMIN = 18
XMAX = 23
YMIN = 18
YMAX = 23
XTEXT = 18.5
YTEXT = 22.5
TEXTSIZE = 24
PLOT_XTICKS = list(range(XMIN+1, XMAX))
PLOT_YTICKS = list(range(YMIN+1, YMAX))
HIST_XTICKS = [-1.0, 0.0, 1.0]
# HIST_XTICKS = [-2.0, -1.0, 0.0, 1.0, 2.0]


def plot_ax(ax, title, mag, true_mag, color, ylabel=False):
    dmag = mag - true_mag

    ax.set_title(title, fontsize=26, fontstyle='italic')
    ax.plot([XMIN, XMAX], [YMIN, YMAX], lw=1, ls="--", color='gray')
    ax.scatter(true_mag, mag, s=6, color=color, alpha=0.15)
    ax.set_xticks(PLOT_XTICKS)
    ax.set_xlabel("Magnitude isolated", fontsize=22)
    if ylabel:
        ax.set_yticks(PLOT_YTICKS)
        ax.set_ylabel("Magnitude blended recovered", fontsize=22)
    else:
        ax.set_yticks([])
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)

    ax.text(
        XTEXT, YTEXT,
        r"$\overline{\Delta_{mag}}$" + " = {0:.3f}".format(np.mean(dmag)),
        fontsize=TEXTSIZE,
    )
    ax.text(
        XTEXT, YTEXT - 0.5,
        r"${\sigma_{mag}}$" + " =  {0:.3f}".format(np.std(dmag)),
        fontsize=TEXTSIZE,
    )

    n = (mag[np.abs(dmag) > 0.75]).shape[0]
    ax.text(
        XTEXT, YTEXT - 1,
        "outliers" + " = {0:2.1f}%".format(n / (dmag).shape[0] * 100),
        fontsize=TEXTSIZE,
    )
    mape = mean_absolute_percentage_error(np.asarray(true_mag), np.asarray(mag))
    ax.text(
        XTEXT, YTEXT - 1.5,
        r"MAPE" + " = {0:2.1f}%".format(mape),
        fontsize=TEXTSIZE,
    )

    in_ax = inset_axes(ax, width="37%", height="37%", loc=4, borderpad=4)

    sns.distplot(
        dmag,
        hist=True,
        kde=True,
        # norm_hist=True,
        bins=int(180 / 5),
        color=color,
        hist_kws={"edgecolor": color},
        kde_kws={"linewidth": 4},
        ax=in_ax,
    )
    plt.title(r"$\Delta_{\rm mag}$", fontsize=24)#, fontsize=26)
    plt.xlim(-2, 2)
    # plt.ylim(0, 5)
    plt.xticks(HIST_XTICKS, fontsize=18)
    plt.yticks([])
    for loc in ['left', 'top', 'right']:
        in_ax.spines[loc].set_visible(True)


def main(datadir):
    data = pd.read_csv(f"{datadir}/output_catalog.csv")

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

    fig5 = plt.figure()
    fig5.set_size_inches(8.75 * 3, 8.75 * 2)

    ax1 = fig5.add_subplot(2, 3, 1)
    plot_ax(ax1, "blend2flux (central)", g1_b2f_mag[g1both], g1_mag[g1both], PALETTE[1], ylabel=True)
    ax2 = fig5.add_subplot(2, 3, 2)
    plot_ax(ax2, "blend2mask2flux (central)", g1_b2m2f_mag[g1both], g1_mag[g1both], PALETTE[1], ylabel=False)
    ax3 = fig5.add_subplot(2, 3, 3)
    plot_ax(ax3, "SExtractor (central)", g1_sextractor_mag[g1both], g1_mag[g1both], PALETTE[1], ylabel=False)
    ax4 = fig5.add_subplot(2, 3, 4)
    plot_ax(ax4, "blend2flux (companion)", g2_b2f_mag[g2both], g2_mag[g2both], PALETTE[5], ylabel=True)
    ax5 = fig5.add_subplot(2, 3, 5)
    plot_ax(ax5, "blend2mask2flux (companion)", g2_b2m2f_mag[g2both], g2_mag[g2both], PALETTE[5], ylabel=False)
    ax6 = fig5.add_subplot(2, 3, 6)
    plot_ax(ax6, "SExtractor (companion)", g2_sextractor_mag[g2both], g2_mag[g2both], PALETTE[5], ylabel=False)

    fig5.subplots_adjust(wspace=0, top=0.95, left=0.05, right=0.99, bottom=0.05, hspace=0.25)
    fig5.savefig("figures/figure5.png")

    # -----------------------------------------------------------------------
    # FIGURE 6
    # --------

    g_b2f_mag_good = np.concatenate([g1_b2f_mag[both], g2_b2f_mag[both]])
    g_b2m2f_mag_good = np.concatenate([g1_b2m2f_mag[both], g2_b2m2f_mag[both]])
    g_sextractor_mag_good = np.concatenate([g1_sextractor_mag[both], g2_sextractor_mag[both]])
    g_mag_good = np.concatenate([g1_mag[both], g2_mag[both]])
    g_b2f_mag_bad = np.concatenate([g1_b2f_mag[central | companion | multiple], g2_b2f_mag[central | companion | multiple]])
    g_b2m2f_mag_bad = np.concatenate([g1_b2m2f_mag[central | companion | multiple], g2_b2m2f_mag[central | companion | multiple]])
    g_mag_bad = np.concatenate([g1_mag[central | companion | multiple], g2_mag[central | companion | multiple]])

    fig6 = plt.figure()
    fig6.set_size_inches(8.75 * 3, 8.75 * 2)

    ax1 = fig6.add_subplot(2, 3, 1)
    plot_ax(ax1, "blend2flux (SEx detected 2 gals)", g_b2f_mag_good, g_mag_good, PALETTE[2], ylabel=True)
    ax2 = fig6.add_subplot(2, 3, 2)
    plot_ax(ax2, "blend2mask2flux (SEx detected 2 gals)", g_b2m2f_mag_good, g_mag_good, PALETTE[2], ylabel=False)
    ax3 = fig6.add_subplot(2, 3, 3)
    plot_ax(ax3, "SExtractor (SEx detected 2 gals)", g_sextractor_mag_good, g_mag_good, PALETTE[2], ylabel=False)
    ax4 = fig6.add_subplot(2, 3, 4)
    plot_ax(ax4, "blend2flux (single or multiple SEx detection)", g_b2f_mag_bad, g_mag_bad, PALETTE[7], ylabel=True)
    ax5 = fig6.add_subplot(2, 3, 5)
    plot_ax(ax5, "blend2mask2flux (single or multiple SEx detection)", g_b2m2f_mag_bad, g_mag_bad, PALETTE[7], ylabel=False)

    fig6.subplots_adjust(wspace=0, top=0.95, left=0.05, right=0.99, bottom=0.05, hspace=0.25)
    fig6.savefig("figures/figure6.png")


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
