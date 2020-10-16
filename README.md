Photometry of blended galaxies with Deep Learning
=================================================

[![License][license-badge]][license-web]
[![arXiv][arxiv-badge]][arxiv-paper]

Code repository aimed at reprooducing the results presented in the [![arXiv][arxiv-badge]][arxiv-paper] paper.

- [Set up](#Set-up)
- [How to cite](#Citing)
- [License](#License)


[license-badge]: https://img.shields.io/badge/license-BSD-blue.svg?style=flat
[license-web]: https://choosealicense.com/licenses/bsd-3-clause/
[arxiv-badge]: https://img.shields.io/badge/arXiv-1905.01324-yellow.svg?style=flat
[arxiv-paper]: https://arxiv.org/abs/1905.01324


Set up
------

### `candels-blender`

The blend-images used in this analysis have been produced with [`candels-blender`][blender].  

1. Install the code and download the individual galaxies from CANDELS (see [instructions][blenderinstall])
2. Choose a seed `<SEED>` and a total number of blend images `<N_BLENDS>` and compute a training set and test set
    ```bash
    candels-blender produce -n <N_BLENDS> --mag_high 23.5 --test_ratio 0.3 --seed <SEED>
    ```
3. Prepare the segmentation labels with 3 channels : [overlap, central galaxy, companion galaxy]
    ```bash
    candels-blender concatenate -d output-s_<SEED>-n_<N_BLENDS> --method ogg_masks
    ```
4. Provide a zeropoint to make the flux conversion for the catalog
    ```bash
    candels-blender convert -d output-s_<SEED>-n_<N_BLENDS> --zeropoint=25.5
    ```

`output-s_<SEED>-n_<N_BLENDS>` therefore becomes the data directory a.k.a. `datadir`s

Set the	environment variable 'COINBLEND_DATADIR' to your chosen	datadir	via
```bash
export COINBLEND_DATADIR=<path-to-datadir>
```

### `coindeblend`

1. Clone this repository
    ```
    git clone https://github.com/aboucaud/coindeblend
    cd coindeblend
    ```

2. Install the required dependencies

  - with `conda`:
    ```
    conda env create -f environment.yml
    conda activate coindeblend
    ```
  - with `pip`:
    ``` 
    python3 -m pip install -r requirements/requirements.txt
    ```

3. Install `coindeblend`

  ```
  python3 -m pip install .
  ```

[deblend]: https://github.com/aboucaud/coindeblend
[blender]: https://github.com/aboucaud/candels-blender
[blenderinstall]: https://github.com/aboucaud/candels-blender#installation

Citing
------
If you use any of this work, please cite the original publication:

```text
@article{10.1093/mnras/stz3056,
    author = {Boucaud, Alexandre and Huertas-Company, Marc and Heneka, Caroline and Ishida, Emille E O and Sedaghat, Nima and de Souza, Rafael S and Moews, Ben and Dole, Hervé and Castellano, Marco and Merlin, Emiliano and Roscani, Valerio and Tramacere, Andrea and Killedar, Madhura and Trindade, Arlindo M M},
    title = "{Photometry of high-redshift blended galaxies using deep learning}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    year = {2019},
    month = {12},
    issn = {0035-8711},
    doi = {10.1093/mnras/stz3056},
    url = {https://doi.org/10.1093/mnras/stz3056},
    note = {stz3056},
    eprint = {http://oup.prod.sis.lan/mnras/advance-article-pdf/doi/10.1093/mnras/stz3056/31176513/stz3056.pdf},
}
```

License
-------
The code is published under the [BSD 3-Clause License](LICENSE). 
