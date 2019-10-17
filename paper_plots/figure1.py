#!/usr/bin/env python
# coding: utf-8
"""
Figure 1
--------

"""
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation

from coindeblend.identity import img_cmap
from coindeblend.identity import gal1_cmap
from coindeblend.visualisation import asin_stretch_norm


SELECTION = {
    'disk': 11,
    'bulge': 1764,
    'bulge + disk': 1298,
    'irregular': 1468
}


def mask_out_pixels(img, seg, segval, noise_factor=1, n_iter=5):
    bseg = binary_dilation(seg, iterations=n_iter)
    centralseg = binary_dilation(np.where(seg == segval, 1, 0),
                                 iterations=n_iter)
    final_mask = np.logical_xor(bseg, centralseg)
    masked_std = np.std(img * np.logical_not(bseg))
    masked_img = img * ~final_mask
    mask_fill = final_mask * np.random.normal(scale=masked_std, size=img.shape)
    noise_map = np.random.normal(scale=masked_std, size=img.shape)
    new_img = masked_img + mask_fill + noise_factor * noise_map

    return new_img.astype(img.dtype)


def plot_line(ax, img, mimg, mseg, title, norm):
    ax[0].set_ylabel(title, fontsize=20)
    ax[0].imshow(img, norm=norm, cmap=img_cmap)
    ax[1].imshow(mimg, norm=norm, cmap=img_cmap)
    ax[2].imshow(~mseg, cmap=gal1_cmap, alpha=0.75)


def main(datadir, log=True):
    # datadir = "../candels-blender/data"
    gals = np.load(os.path.join(datadir, "candels_img.npy"))
    segs = np.load(os.path.join(datadir, "candels_seg.npy"))

    imgs = [gals[i] for i in SELECTION.values()]
    masks = [segs[i] for i in SELECTION.values()]
    mimgs = [mask_out_pixels(img, seg, seg[64, 64]) for img, seg in zip(imgs, masks)]
    msegs = [np.where(seg == seg[64, 64], 1, 0).astype(bool) for seg in masks]

    norm = asin_stretch_norm(imgs + mimgs)

    fig, axes = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True, figsize=(15, 20), tight_layout=True)

    for idx, galtype in enumerate(SELECTION.keys()):
        plot_line(axes[idx], imgs[idx], mimgs[idx], msegs[idx], galtype, norm)

    fig.savefig('plots/figure1.png')


if __name__ == "__main__":
    if not os.path.exists('plots'):
        os.makedirs('plots')

    import sys
    try:
        datadir = sys.argv[1]
    except:
        sys.exit(
            "Please indicate the relative path to the data directory.\n"
            f"python {__name__} ../relative/path/to/CANDELS/data/"
        )

    main(datadir)
