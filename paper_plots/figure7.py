#!/usr/bin/env python
# coding: utf-8
"""
Figure 7
--------
Error in predicted magnitude as a function of magnitude difference
between the two blended galaxies

"""
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as clr
import seaborn as sns
import pandas as pd
from scipy.stats import binned_statistic

from deblend.identity import paper_palette as PALETTE


sns.set_palette(PALETTE)
sns.set_context("paper")
sns.set_style("ticks")

plt.rc("axes.spines", top=False, right=False)
plt.rc("font", size=20)
plt.rc("xtick", labelsize='large')
plt.rc("ytick", labelsize='large')

cmap1 = clr.LinearSegmentedColormap.from_list("my_blue", ((0.9, 0.9, 0.9), PALETTE[1]), N=4)
cmap2 = clr.LinearSegmentedColormap.from_list(
    "my_cdbqf", ((0.9, 0.9, 0.9), PALETTE[5]), N=4
)

XMIN = -2
XMAX = 2
YMIN_DL = -0.7
YMAX_DL = 0.7
YMIN_SEX = -3.2
YMAX_SEX = 3.2

SEX_TICKS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
DL_TICKS = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
MAG_TICKS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

ZP_SEX = 25.67
ZP_NN = 25.96


data = pd.read_csv("output_catalog.csv")

dis = data["distance"]
g1_rad = data["g1_rad"]
g2_rad = data["g2_rad"]
g1_morph = data["g1_type"]
g2_morph = data["g2_type"]
g1_mag = data["g1_mag"]
g2_mag = data["g2_mag"]

delta_mag = g1_mag - g2_mag

g1_sextractor_mag = flux2mag(data["g1_flux_sex"], ZP_SEX)
g2_sextractor_mag = flux2mag(data["g2_flux_sex"], ZP_SEX)

g1_b2f_mag = flux2mag(data["g1_flux_b2f"], ZP_NN)
g2_b2f_mag = flux2mag(data["g2_flux_b2f"], ZP_NN)

g1_b2m2f_mag = flux2mag(data["g1_flux_b2m2f"], ZP_NN)
g2_b2m2f_mag = flux2mag(data["g2_flux_b2m2f"], ZP_NN)

isnan = np.isnan(g1_sex_mag)
isnan2 = np.isnan(g2_sex_mag)

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

bin_means_b2f_c, bin_edges, binnumber = binned_statistic(
    delta_mag, g1_b2f_mag - g1_mag, statistic="median", bins=6
)
bin_std_b2f_c, bin_edges, binnumber = binned_statistic(
    delta_mag, g1_b2f_mag - g1_mag, statistic=np.std, bins=6
)

bin_means_b2f_c_sph, bin_edges, binnumber = binned_statistic(
    delta_mag[g1_morph == "sph"],
    g1_b2f_mag[g1_morph == "sph"] - g1_mag[g1_morph == "sph"],
    statistic="median",
    bins=6,
)
bin_std_b2f_c_sph, bin_edges, binnumber = binned_statistic(
    delta_mag[g1_morph == "sph"],
    g1_b2f_mag[g1_morph == "sph"] - g1_mag[g1_morph == "sph"],
    statistic=np.std,
    bins=6,
)

bin_means_b2f_c_sphd, bin_edges, binnumber = binned_statistic(
    delta_mag[g1_morph == "sphd"],
    g1_b2f_mag[g1_morph == "sphd"] - g1_mag[g1_morph == "sphd"],
    statistic="median",
    bins=6,
)
bin_std_b2f_c_sphd, bin_edges, binnumber = binned_statistic(
    delta_mag[g1_morph == "sphd"],
    g1_b2f_mag[g1_morph == "sphd"] - g1_mag[g1_morph == "sphd"],
    statistic=np.std,
    bins=6,
)

bin_means_b2f_c_disk, bin_edges, binnumber = binned_statistic(
    delta_mag[g1_morph == "disk"],
    g1_b2f_mag[g1_morph == "disk"] - g1_mag[g1_morph == "disk"],
    statistic="median",
    bins=6,
)
bin_std_b2f_c_disk, bin_edges, binnumber = binned_statistic(
    delta_mag[g1_morph == "disk"],
    g1_b2f_mag[g1_morph == "disk"] - g1_mag[g1_morph == "disk"],
    statistic=np.std,
    bins=6,
)

bin_means_b2f_c_irr, bin_edges, binnumber = binned_statistic(
    delta_mag[g1_morph == "irr"],
    g1_b2f_mag[g1_morph == "irr"] - g1_mag[g1_morph == "irr"],
    statistic="median",
    bins=6,
)
bin_std_b2f_c_irr, bin_edges, binnumber = binned_statistic(
    delta_mag[g1_morph == "irr"],
    g1_b2f_mag[g1_morph == "irr"] - g1_mag[g1_morph == "irr"],
    statistic=np.std,
    bins=6,
)

data_bin = [[] for i in range(len(bin_edges) - 1)]
data_bin_b2m2f = [[] for i in range(len(bin_edges) - 1)]
data_bin_sex = [[] for i in range(len(bin_edges) - 1)]
bin_bool = []
data_bin2 = [[] for i in range(len(bin_edges) - 1)]
data_bin_b2m2f2 = [[] for i in range(len(bin_edges) - 1)]
data_bin_sex2 = [[] for i in range(len(bin_edges) - 1)]
for j in range(len(bin_edges) - 1):
    for i in range(data.shape[0]):
        if bin_edges[j] <= delta_mag[i] < bin_edges[j + 1]:
            data_bin[j].append(g1_b2f_mag[i] - g1_mag[i])
            bin_bool.append(True)
            data_bin_b2m2f[j].append(g1_b2m2f_mag[i] - g1_mag[i])

            if isnan[i] == False:
                data_bin_sex[j].append(g1_sex_mag[i] - g1_mag[i])

        if bin_edges[j] <= -delta_mag[i] < bin_edges[j + 1]:
            data_bin2[j].append(g2_b2f_mag[i] - g2_mag[i])
            data_bin_b2m2f2[j].append(g2_b2m2f_mag[i] - g2_mag[i])
            if isnan2[i] == False:
                data_bin_sex2[j].append(g2_sex_mag[i] - g2_mag[i])




BINS_POSITION = [
    np.mean([bin_edges[j], bin_edges[j + 1]]) for j in range(len(bin_edges) - 1)
]
distance = dis / g2_rad

def plot_ax(ax, title, mag_diff, dmag, data_bin, cut, cmap, cbar_kw={}, ylabel=False, sex=False, companion=False):
    # if sex:
        # mag_diff = mag_diff[~isnan2]
        # dmag = dmag[~isnan2]
        # color = distance[~isnan2]

    if companion:
        mag_diff = -mag_diff
        xlabel = r"$mag_{companion}-mag_{central}$"
        color = dis / g2_rad
    else:
        xlabel = r"$mag_{central}-mag_{companion}$"
        color = dis / g1_rad

    mag_diff = mag_diff[cut]
    dmag = dmag[cut]
    color = color[cut]


    bp = ax.boxplot(
        data_bin,
        positions=BINS_POSITION,
        showfliers=False,
        labels=[item for item in range(len(BINS_POSITION))],
        whis=[5, 95],
    )
    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[element], lw=3, alpha=0.35)

    # for box in bp["boxes"]:
        # box.set(color='#eeeeee', lw=5, alpha=1)

    ax.set_title(title, fontsize=26, fontstyle='italic')

    data = ax.scatter(
        mag_diff, dmag,
        # s=10,
        s=4,
        vmin=0, vmax=4,
        c=color,
        cmap=cmap, zorder=-1)
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks(MAG_TICKS)
    ax.set_xticklabels(MAG_TICKS)
    ax.tick_params(axis="x", which="major", pad=5)
    ax.axhline(0.0, lw=2, ls='--', color='gray')

    ax.set_xlabel(xlabel, fontsize=22)
    if sex:
        ax.set_yticks(SEX_TICKS)
        ax.set_ylim(YMIN_SEX, YMAX_SEX)
    else:
        ax.set_yticks(DL_TICKS)
        ax.set_ylim(YMIN_DL, YMAX_DL)
    if ylabel:
        ax.set_ylabel(r"$\Delta_{\rm mag}$", fontsize=22)

    return data


def main(datadir):

    fig, axes = plt.subplots(2, 3, constrained_layout=True)
    fig.set_size_inches(8.75 * 3, 8.75 * 2)

    plot_ax(axes[0, 0], "blend2flux (central)", delta_mag, g1_b2f_mag - g1_mag, data_bin, g1both, cmap1, ylabel=True, sex=False)
    plot_ax(axes[0, 1], "blend2mask2flux (central)", delta_mag, g1_b2m2f_mag - g1_mag, data_bin_b2m2f, g1both, cmap1, ylabel=False, sex=False)
    data_g1 = plot_ax(axes[0, 2], "SExtractor (central)", delta_mag, g1_sex_mag - g1_mag, data_bin_sex, g1both, cmap1, ylabel=False, sex=True)
    cbar1 = fig.colorbar(data_g1, ax=[axes[0, 2]])
    cbar1.ax.set_ylabel(r"distance / $R_e$", rotation=90, va="top", fontsize=22)
    cbar1.set_ticks([0, 1, 2, 3, 4])
    plot_ax(axes[1, 0], "blend2flux (companion)", delta_mag, g2_b2f_mag - g2_mag, data_bin2, g1both, cmap2, ylabel=True, sex=False, companion=True)
    plot_ax(axes[1, 1], "blend2mask2flux (companion)", delta_mag, g2_b2m2f_mag - g2_mag, data_bin_b2m2f2, g1both, cmap2, ylabel=False, sex=False, companion=True)
    data_g2 = plot_ax(axes[1, 2], "SExtractor (companion)", delta_mag, g2_sex_mag - g2_mag, data_bin_sex2, g1both, cmap2, ylabel=False, sex=True, companion=True)
    cbar2 = fig.colorbar(data_g2, ax=[axes[1, 2]])
    cbar2.ax.set_ylabel(r"distance / $R_e$", rotation=90, va="top", fontsize=22)
    cbar2.set_ticks([0, 1, 2, 3, 4])

    plt.savefig("figures/figure7.png", dpi=70)


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
