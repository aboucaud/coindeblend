#!/usr/bin/env python
# coding: utf-8
"""
Figure #2
---------
Blend images - distance vs. magnitude difference

"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows

from coindeblend.identity import img_cmap
from coindeblend.visualisation import asin_stretch_norm

DIST_BINS = [(2,8), (8, 13), (13, 18), (18, 60)]
MAGDIFF_BINS = [(0, 0.35), (0.35, 0.76), (0.76, 1.25), (1.25, 2)]


def plot_blends(blends, indices, log):
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(20, 20), tight_layout=True)

    norm = asin_stretch_norm(blends[indices])

    for i, ax in enumerate(axes.flatten()):
        idx = indices[i]
        ax.imshow(blends[idx], norm=norm, cmap=img_cmap)
        # ax.set_title(f"ID={idx}, dist={small_cat.distance[idx]:.2f}, magdiff={small_cat.magdiff[idx]:.2f}")
        ax.set_axis_off()

    a = AnchoredDirectionArrows(
        axes[0,0].transAxes, r'$\Delta$ mag', 'distance [pix]', loc='upper left',
        aspect_ratio=-1,
        sep_x=0.02, sep_y=-0.04,
        color='white'
    )
    axes[0, 0].add_artist(a)

    fig.savefig(f"plots/figure2.png")


def main(datadir):
    blends_catalog = pd.read_csv(os.path.join(datadir, 'test_blend_cat.csv'))
    blends_images = np.load(os.path.join(datadir, "test_images.npy"), mmap_mode='r')

    blends_catalog['magdiff'] = np.abs(blends_catalog.g1_mag - blends_catalog.g2_mag)
    small_cat = blends_catalog[['distance', 'magdiff']]

    # For each distance/mag_diff bin, randomly select one blend in the catalog
    blend_indices = []
    for dmin, dmax in DIST_BINS:
        for mmin, mmax in MAGDIFF_BINS:
            x = small_cat.copy()
            x = x[x.distance < dmax]
            x = x[x.distance > dmin]
            x = x[x.magdiff < mmax]
            x = x[x.magdiff > mmin]
            blend_indices.append(np.random.choice(x.index.values))

    plot_blends(blends_images, blend_indices, log=True)


if __name__ == "__main__":
    if not os.path.exists('plots'):
        os.makedirs('plots')

    import sys
    try:
        datadir = sys.argv[1]
    except:
        sys.exit(
            "Please indicate the relative path to the data directory.\n"
            f"python {__name__} ../relative/path/to/data/"
        )

    main(datadir)
