import os
import sep
import numpy as np
import pylab as plt
import pandas as pd

X_CENTER = 63.5
Y_CENTER = 63.5
ZP_SEXTRACTOR = 25.67
SEXCONFIG = {
    "hot": {
        "final_area": 6,
        "final_threshold": 4,
        "final_cont": 0.0001,
        "final_nthresh": 64,
    },
    "cold": {
        "final_area": 10,
        "final_threshold": 5,
        "final_cont": 0.01,
        "final_nthresh": 64,
    },
    "mine": {
        "final_area": 10,
        "final_threshold": 4,
        "final_cont": 0.0001,
        "final_nthresh": 64,
    }
}

def run_sextractor(image, background, config):
    return sep.extract(
        image,
        config["final_threshold"],
        err=background.globalrms,
        minarea=config["final_area"],
        deblend_nthresh=config["final_nthresh"],
        deblend_cont=config["final_cont"],
        segmentation_map=True,
    )

def plot_galaxy_with_markers(idx, image, segmap, gal_cen, gal_comp=None):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.scatter(gal_cen["x"], gal_cen["y"], marker="x", color="red")
    if gal_comp is not None:
        plt.scatter(gal_comp["x"], gal_comp["y"], marker="o")
    plt.subplot(1, 2, 2)
    plt.imshow(segmap)
    plt.savefig(f"plots_segmap/{idx}.png")
    plt.close()


def analyse_single_blend(idx, blend_image, cat, fig=False):
    # Compute the background
    bkg = sep.Background(blend_image)

    # Run detection with the 'cold' strategy
    source, segmap = run_sextractor(blend_image, bkg, SEXCONFIG["cold"])
    n_detect_cold = len(source["x"])
    n_detect_hot = 0

    if n_detect_cold < 2:
        # Rerun SExtractor with  the 'hot' stratefy
        source, segmap = run_sextractor(blend_image, bkg, SEXCONFIG["hot"])
        n_detect_hot = len(source["x"])

    n_detections = max(n_detect_cold, n_detect_hot)

    if n_detections == 0:
        return {}

    positions = np.hypot(
        source["x"] - X_CENTER,
        source["y"] - Y_CENTER
    )
    indx = positions.argsort().tolist()
    id_central = indx.pop(0)

    result = {}
    result["n_sources"] = n_detections

    if n_detections >= 2:
        # Once the central galaxy is found, order the remaining galaxies by flux
        flux_indx = np.argsort(source[indx]["flux"])
        id_companion = flux_indx[-1]

        result["flux_central"] = source[id_central]["flux"]
        result["flux_companion"] = source[id_companion]["flux"]

        if fig:
            plot_galaxy_with_markers(idx, blend_image, segmap,
                                     source[id_central], source[id_companion])
    else:
        # If a single object is detected even with the 'hot' strategy
        # assign that galaxy to the central or the companion depending
        # on the relative distance to it

        # x_companion = X_CENTER + full_cat["shift_x"][idx]
        # y_companion = Y_CENTER + full_cat["shift_y"][idx]

        # dist_to_companion = np.hypot(
        #     source["x"][0] - x_companion,
        #     source["y"][0] - y_companion
        # )
        dist_to_center = np.hypot(
            source["x"][0] - X_CENTER,
            source["y"][0] - Y_CENTER
        )

        # if dist_to_companion > dist_to_center:
        if dist_to_center < 0.5 * cat["g1_rad"][idx]:
            result["flux_central"] = source[id_central]["flux"]
        else:
            result["flux_companion"] = source[id_central]["flux"]

    return result


def analyse_blends(catalog, fig):
    datadir = os.getenv("COINBLEND_DATADIR")

    data_blends = np.load(f"{datadir}/{sample}_images.npy")
    fluxes_cat = np.load(f"{datadir}/{sample}_flux.npy")

    n_sources = np.empty(len(data_blends), dtype=np.uint8)
    flux_central_estimated = np.empty(len(data_blends), dtype=np.float)
    flux_companion_estimated = np.empty(len(data_blends), dtype=np.float)

    for id_blend, image in enumerate(data_blends):
        result = analyse_single_blend(id_blend, image, catalog, fig=fig)

        flux_central_estimated[id_blend] = result.get("flux_central", np.nan)
        flux_companion_estimated[id_blend] = result.get("flux_companion", np.nan)
        n_sources[id_blend] = result.get("n_sources", 0)

    return n_sources, flux_central_estimated, flux_companion_estimated


def main(version, sample, fig):
    cat = pd.read_csv(f"{datadir}/{sample}_catalogue.csv")

    g1_true_mag = cat["g1_mag"]
    g2_true_mag = cat["g2_mag"]
    delta_true_mag = g1_true_mag - g2_true_mag

    n_sources, g1_sex, g2_sex = analyse_blends(cat, fig)

    flux_perc_central_estimated = (
        g1_sex / (g1_sex + g2_sex)
    )
    # flux_perc_central_catalog = fluxes_cat[:, 0] / fluxes_cat.sum(axis=1)
    g1_sex_mag = -2.5 * np.log10(g1_sex) + ZP_SEXTRACTOR
    g2_sex_mag = -2.5 * np.log10(g2_sex) + ZP_SEXTRACTOR

    # In Python a == a is False for NaN values
    g1_detected = g1_sex == g1_sex
    g2_detected = g2_sex == g2_sex
    at_least_two_sources_detected = g1_detected & g2_detected
    two_sources_detected = n_sources == 2
    three_and_more = n_sources > 2

    print(f"Number of detected g1: {g1_detected.sum()}")
    print(f"Number of detected g2: {g2_detected.sum()}")
    print(f"Number of 2+ gals detected: {at_least_two_sources_detected.sum()}")
    print(f"Number of 2 gals detected: {two_sources_detected.sum()}")
    print(f"Number of 3+ gals detected: {three_and_more.sum()}")

    residual_g1_mag = g1_sex_mag - g1_true_mag
    residual_g2_mag = g2_sex_mag - g2_true_mag

    plt.scatter(delta_true_mag[at_least_two_sources_detected], residual_g1_mag[at_least_two_sources_detected], s=1)
    plt.scatter(delta_true_mag[three_and_more], residual_g1_mag[three_and_more], marker="o")
    plt.show()

    # fig1 = f"{sample}/residuals_perc_conf_{version}_{sample}.png"
    # fig2 = f"{sample}/residual_flux_{version}_{sample}.png"

    # output_file = f"{sample}/survived_sextractor_{sample}_{version}.txt"
    # output_flux_file = f"{sample}/flux_sextractor_{sample}_{version}.txt"


if __name__ == "__main__":
    version = "afterscreening"
    sample = "test"
    fig = False

    main(version, sample, fig)
