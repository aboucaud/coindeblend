from astropy.visualization import ImageNormalize
from astropy.visualization import MinMaxInterval
from astropy.visualization import LogStretch
from astropy.visualization import AsinhStretch


def asin_stretch_norm(images):
    return ImageNormalize(
        images,
        interval=MinMaxInterval(),
        stretch=AsinhStretch(),
    )


def log_stretch_norm(images):
    return ImageNormalize(
        images,
        interval=MinMaxInterval(),
        stretch=LogStretch(),
    )
