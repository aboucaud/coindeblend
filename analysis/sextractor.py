import sep
import numpy as np
import pylab as plt
import pandas as pd

# parameters
version = 'afterscreening'
sample = 'test'
mix = 'irr'
fig = False

DATADIR = "/Users/alexandre/work/lsst/deblending/coin/candels-blender/output-s_666-n_30000"
CATDIR = "/Users/alexandre/work/lsst/deblending/coin/candels-blender/output-s_666-n_30000"
FINALCATDIR = "/Users/alexandre/work/lsst/deblending/coin/catalog_results"

output_file = sample + '/survived_sextractor_'  + sample + '_' + version + '_' + mix + '.txt'
output_flux_file = sample + '/flux_sextractor_'  + sample + '_' + version + '_' + mix + '.txt'

# path1 = 'test_blend_cat_emille.csv'
# full_cat = pd.read_csv(path1)


# path2 = '../../plots/final/flux_catalog_new_dataset_after_screening.csv'
# cat = pd.read_csv(path2)

full_cat = pd.read_csv(f"{CATDIR}/test_blend_cat_emille.csv")
cat = pd.read_csv(f"{FINALCATDIR}/flux_catalog_new_dataset_after_screening.csv")

g1_mag = cat["g1_mag"]
g2_mag = cat["g2_mag"]

delta_mag = g1_mag-g2_mag

status = {}
status['hot'] = {'final_area': 6,
                  'final_threshold': 4,
                  'final_cont': 0.0001,
                  'final_nthresh': 64}

status['cold'] = {'final_area': 10,
                  'final_threshold': 5,
                  'final_cont': 0.01,
                  'final_nthresh': 64}

status['mine'] = {'final_area': 10,
                  'final_threshold': 4,
                  'final_cont': 0.0001,
                  'final_nthresh': 64}


fig1 = sample + '/residuals_perc_conf' + str(version)+ '_' + sample + '_' + mix + '.png'
fig2 = sample + '/residual_flux_' + str(version) + '_' + sample + '_' + mix + '.png'

if mix == 'irr':
    # load blended images
    data_blends = np.load(f"{DATADIR}/{sample}_images.npy")

    # load fluxes
    fluxes_cat = np.load(f"{DATADIR}/{sample}_flux.npy")


else:
    # load blended images
    data_blends = np.load("../data/" + sample + "_images.npy")

    # load fluxes
    fluxes_cat = np.load('../data/' + sample + '_flux.npy')


# calculate fluxes
flux_central_estimated = []
flux_companion_estimated = []
flux_perc_central_estimated = []
flux_perc_central_catalog = []
nsources = []
surv = []
coord = []

nsources=[0]


for j in range(data_blends.shape[0]):
#while nsources[-1] < 3:
    #print(j)

    # get background
    bkg = sep.Background(data_blends[j])

    # get sources
    source, segmap = sep.extract(data_blends[j], status['cold']['final_threshold'],
                         err=bkg.globalrms, minarea=status['cold']['final_area'],
                         deblend_nthresh=status['cold']['final_nthresh'], deblend_cont=status['cold']['final_cont'],
                         segmentation_map=True)

    # store number of detected sources
    nsources.append(source['x'].shape[0])

    if nsources[-1] > 1:

        positions = [np.sqrt(pow(source[i]['x'] - 63.5, 2) + pow(source[i]['y'] - 63.5, 2)) for i in range(nsources[-1])]
        indx = np.array(positions).argsort()

        flux_central_estimated.append(source[indx[0]]['flux'])
        coord.append([source[indx[0]]['x'], source[indx[0]]['y']])

        flux_indx = np.argsort(source[:]['flux'])

        if flux_indx[-1] != indx[0]:
            num = flux_indx[-1]
        else:
            num = flux_indx[-2]

        flux_companion_estimated.append(source[num]['flux'])
        flux_perc_central_estimated.append(source[indx[0]]['flux']/(source[indx[0]]['flux']+source[num]['flux']))
        flux_perc_central_catalog.append(fluxes_cat[j][0]/sum(fluxes_cat[j]))
        surv.append(int(j))

        if fig:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(data_blends[j])
            plt.scatter(source[indx[0]]['x'], source[indx[0]]['y'], marker = 'x', color='red')
            plt.scatter(source[num]['x'], source[num]['y'], marker='o')

            plt.subplot(1,2,2)
            plt.imshow(segmap)
            plt.savefig('plots_segmap/' + str(j) + '.png')

    else:
        # get sources
        source, segmap = sep.extract(data_blends[j], status['hot']['final_threshold'],
                             err=bkg.globalrms, minarea=status['hot']['final_area'],
                             deblend_nthresh=status['hot']['final_nthresh'], deblend_cont=status['hot']['final_cont'], segmentation_map=True)

        # store number of detected sources
        nsources.append(source['x'].shape[0])

        if nsources[-1] > 0:
            positions = [np.sqrt(pow(source[i]['x'] - 63.5, 2) + pow(source[i]['y'] - 63.5, 2)) for i in range(nsources[-1])]
            indx = np.array(positions).argsort()

            coord.append([source[indx[0]]['x'], source[indx[0]]['y']])

            if fig:
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(data_blends[j])
                plt.scatter(source[indx[0]]['x'], source[indx[0]]['y'], marker = 'x', color='red')


            if nsources[-1] > 1:

                flux_indx = np.argsort(source[:]['flux'])
                if flux_indx[-1] != indx[0]:
                    num = flux_indx[-1]
                else:
                    num = flux_indx[-2]

                flux_central_estimated.append(source[indx[0]]['flux'])
                flux_companion_estimated.append(source[num]['flux'])
                flux_perc_central_estimated.append(source[indx[0]]['flux']/(source[indx[0]]['flux']+source[num]['flux']))
                flux_perc_central_catalog.append(fluxes_cat[j][0]/sum(fluxes_cat[j]))
                surv.append(int(j))

                if fig:
                    plt.scatter(source[num]['x'], source[num]['y'], marker='o')
                    plt.subplot(1,2,2)
                    plt.imshow(segmap)
                    plt.savefig('plots_segmap/' + str(j) + '.png')

            else:
                if fig:
                    plt.subplot(1,2,2)
                    plt.imshow(segmap)
                    plt.savefig('plots_segmap/' + str(j) + '.png')

                dist_companion = np.sqrt(pow(full_cat['shift_x'][j] - source['x'][0], 2) + pow(full_cat['shift_y'][j] - source['y'][0], 2))
                dist_central = np.sqrt(pow(source['x'][0] - 63.5,2) + pow(source['y'][0] - 63.5,2))

                if dist_companion > dist_central:
                    flux_central_estimated.append(source[indx[0]]['flux'])
                    flux_companion_estimated.append(-99)
                else:
                    flux_companion_estimated.append(source[indx[0]]['flux'])
                    flux_central_estimated.append(-99)

                flux_perc_central_estimated.append(-99)
                flux_perc_central_catalog.append(-99)

        else:
            if fig:
                plt.subplot(1,2,2)
                plt.imshow(segmap)
                plt.savefig('plots_segmap/' + str(j) + '.png')

            flux_central_estimated.append(-99)
            flux_perc_central_estimated.append(-99)
            flux_perc_central_catalog.append(-99)
            flux_companion_estimated.append(-99)

    if fig:
        plt.close('all')


flux_central_estimated = np.array(flux_central_estimated)
flux_companion_estimated = np.array(flux_companion_estimated)

isnan = flux_central_estimated == -99
isnan2 = flux_companion_estimated == -99
both_galaxies_detected = ~isnan & ~isnan2
# isnan2 = np.array([False if item==-99 else True for item in flux_companion_estimated ])
print(f"Number of detected g1: {(~isnan).sum()}")
print(f"Number of detected g2: {(~isnan2).sum()}")
print(f"Number of both detected: {both_galaxies_detected.sum()}")

zp_sex = 25.67
g1_sex_mag = -2.5*np.log10(flux_central_estimated[~isnan])+zp_sex
g2_sex_mag = -2.5*np.log10(flux_companion_estimated[~isnan])+zp_sex


plt.scatter(delta_mag[~isnan], g1_sex_mag[~isnan]-g1_mag[~isnan], s=1)
plt.show()

g1_cut = (g1_sex_mag[~isnan]-g1_mag[~isnan]) < -1
cut_blends = data_blends[g1_cut]

indx = np.where(g1_cut)[0]


def move_wrong_masks(indices):
    import pathlib
    import shutil

    p = pathlib.Path.cwd()
    plot_dir = p / "plots_segmap"
    newdir = p / "plots_tocheck"
    if not newdir.exists():
        newdir.mkdir()

    tmp_file = "{}.png"

    for val in indices:

        fil = tmp_file.format(val)
        src = plot_dir / fil
        dst = newdir / fil
        shutil.copy(src, dst)



"""
plt.figure()
plt.hist(flux_central_estimated[surv]-fluxes_cat[surv][:,0], label='central', alpha=0.7, bins=100)
plt.hist(flux_companion_estimated[surv] - fluxes_cat[surv][:,1], label='companion', alpha=0.7, bins=100)
plt.legend()
plt.xlim(-100,100)
plt.xlabel('flux_estimated - true_flux')
plt.savefig('hist_comp_central.png')


res_perc = np.array(flux_perc_central_estimated)[surv] - np.array(flux_perc_central_catalog)[surv]
res_central = np.array(flux_central_estimated)[surv] - fluxes_cat[surv][:,0]

result_number1 = np.mean([res_central[i]/fluxes_cat[surv][:,0][i] for i in range(len(fluxes_cat[surv]))])
result_number2 = np.mean([(np.array(flux_companion_estimated)[surv][i] - fluxes_cat[surv][:,1][i])/fluxes_cat[surv][:,1][i] for i in range(len(fluxes_cat[surv]))])

print('Result: ' + str(np.mean([result_number1, result_number2])))

plt.figure()
h1 = plt.hist(res_perc, bins=50)
plt.text(-0.65, 0.95*max(h1[0]), 'mean: ' + str(round(100*np.mean(res_perc),2)) + ' +/- ' + str(round(100*np.std(res_perc),2)))
#plt.text(-0.65, 0.7*max(h1[0]), 'minarea:      ' + str(int(final_area)) + '\nthreshold:    ' + str(int(final_threshold)) + '\ncont:            ' + str(round(final_cont,4)) + '\nnthresh:      '+ str(int(final_nthresh)) + '\nnon-detections: ' + str(round(100*(data_blends.shape[0]-len(surv))/float(data_blends.shape[0]))) + '%')
plt.xlabel('% flux from central galaxy (estimated - catalog)')
plt.ylabel('Number of galaxies')
plt.savefig(fig1)

plt.figure()
h2 = plt.hist(res_central, bins=50)
plt.xlabel('central galaxy flux (estimated - catalog)')
plt.ylabel('Number of galaxies')
plt.text(200, 0.95*max(h2[0]), 'mean: ' + str(round(np.mean(res_central),2)) + ' +/- ' + str(round(np.std(res_central),2)))
#plt.text(200, 0.7*max(h2[0]), 'minarea:      ' + str(int(final_area)) + '\nthreshold:    ' + str(int(final_threshold)) + '\ncont:            ' + str(round(final_cont,4)) + '\nnthresh:      '+ str(int(final_nthresh)) + '\nnon-detections: ' + str(round(100*(data_blends.shape[0]-len(surv))/float(data_blends.shape[0]))) + '%')
plt.savefig(fig2)


op1 = open(output_file, 'w')
for item in surv:
    op1.write(str(int(item)) + '\n')
op1.close()

op2 = open(output_flux_file, 'w')
for i in range(len(flux_central_estimated)):
    op2.write(str(flux_central_estimated[i]) + ',' + str(flux_companion_estimated[i]) + '\n')
op2.close()

"""
