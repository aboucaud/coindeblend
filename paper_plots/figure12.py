#!/usr/bin/env python
# coding: utf-8
"""
Figure 12
---------
Segmentation metrics IoU and comparison between pure segmentation network (UNet)
and result after optimising for the flux (blend2mask2flux)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from coindeblend.identity import paper_palette as PALETTE
from coindeblend.scores import iou_bitmap

plt.rc("font", size=15)
plt.rc("xtick", labelsize='medium')
plt.rc("ytick", labelsize='medium')


def compute_iou(pred_file):
    y_pred_raw = np.load(pred_file)
    y_pred = y_pred_raw.round()
    iou_list = [[iou_bitmap(a, b)
                 for a, b in zip(yp.T, yt.T)]
                for yp, yt in zip(y_pred[cat.id], y_test[cat.id])]
    iou_array = np.asarray(iou_list)
    iou_array = iou_array[:, 1:]
    return iou_array


def main(datadir, predictiondir):
    y_test = np.load(f'{datadir}/test_masks.npy', mmap_mode='r')
    cat = pd.read_csv(f'{datadir}/test_catalog.csv')
    cat.reset_index(drop=True, inplace=True)

    pred_file_seg = f'{predictiondir}/y_pred-d5_f32_bce.npy'
    pred_file_b2m2f = f'{predictiondir}/blend2mask2flux_main_full7irrClean-Mix.1-dropTrain-test_with_irr-mask_results.npy'

    cat['dmag'] = np.abs(cat.g1_mag - cat.g2_mag)

    iou_seg = compute_iou(pred_file_seg)
    cat['iou_seg'] = np.mean(iou_seg, axis=1)
    cat['iou_g1_seg'] = iou_seg[:, 0]
    cat['iou_g2_seg'] = iou_seg[:, 1]

    iou_b2m2f = compute_iou(pred_file_b2m2f)
    cat['iou_b2m2f'] = np.mean(iou_b2m2f, axis=1)
    cat['iou_g1_b2m2f'] = iou_b2m2f[:, 0]
    cat['iou_g2_b2m2f'] = iou_b2m2f[:, 1]

    cat["type"] = cat.g1_type + '-' + cat.g2_type
    cat["type"].loc[cat['type'] == 'sphd-disk'] = 'disk-sphd'
    cat["type"].loc[cat['type'] == 'sph-disk'] = 'disk-sph'
    cat["type"].loc[cat['type'] == 'sphd-sph'] = 'sph-sphd'
    cat["type"].loc[cat['type'] == 'irr-disk'] = 'disk-irr'
    cat["type"].loc[cat['type'] == 'irr-sph'] = 'sph-irr'
    cat["type"].loc[cat['type'] == 'irr-sphd'] = 'sphd-irr'


    # -------------------------------------------------------------------------------
    # FIGURE 1 - HISTOGRAMS
    # ---------------------
    ax = (cat
    [["iou_seg", "iou_b2m2f"]]
    .plot
    .hist(bins=50, normed=True, alpha=.5, color=[palette[2], palette[7]]))
    ax.legend(['UNet', 'blend2mask2flux'], loc=2, fontsize='small', frameon=False)
    plt.xlim(0,1)
    plt.xlabel('IoU', fontsize=15)
    plt.ylabel('Normalised counts', fontsize=15)
    sns.despine()
    plt.tight_layout()
    plt.savefig('figures/figure12-1.png')

    # -------------------------------------------------------------------------------
    # FIGURE 2 - MORPHOLOGY
    # ---------------------
    ax = (cat
    .groupby('type')
    .mean()
    [["iou_seg", "iou_b2m2f"]]
    .plot
    .barh(width=0.7, alpha=.5, color=[palette[2], palette[7]]))
    ax.legend(['UNet', 'blend2mask2flux'], loc=2, fontsize='small', framealpha=None)
    plt.xlim(0,1)
    plt.xlabel('IoU', fontsize=15)
    plt.ylabel('Galaxy types', fontsize=15)
    sns.despine()
    plt.tight_layout()
    plt.savefig('figures/figure12-2.png')


if __name__ == "__main__":
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')

    import sys
    try:
        datadir = sys.argv[1]
        predictdir = sys.argv[2]
    except:
        sys.exit(
            "Please indicate the relative path to the data directory.\n"
            f"python {__name__} ../relative/path/to/data/"
        )

    main(datadir, predictdir)
