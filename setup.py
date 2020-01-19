from setuptools import find_packages, setup

setup(
    name="coindeblend",
    version="0.1.0",
    description="Deep learning toolbox for galaxy segmentation and deblending",
    author="Alexandre Boucaud",
    author_email="aboucaud@apc.in2p3.fr",
    packages=find_packages(),
    license="BSD",
    install_requires = [
        "numpy",
        "tensorflow<2.0",
        "keras",
        "h5py",
        "scikit-learn",
        "scipy",
        "pandas",
        "matplotlib",
        "sep",
    ],
    python_requires='>=3.6',
)
