"""
Python port of ffdmRead.m
Read FFDM image from DICOM file and normalize intensity.
"""
import numpy as np
import pydicom


def ffdmRead(impath, info):
    """
    Read FFDM image from DICOM file.

    Parameters
    ----------
    impath : str
        Path to DICOM file.
    info : dict
        Structure as returned by getinfo().

    Returns
    -------
    imn : ndarray
        Normalized image in [0, 1].
    im : ndarray
        Raw image with original intensity values.
    """
    ds = pydicom.dcmread(impath)
    im = ds.pixel_array.astype(np.float32)

    # Determine vendor
    vendor = (info.get('source') or '').upper()
    is_negative = ('FUJIFILM' in vendor or 'SECTRA' in vendor or 'PHILIPS' in vendor)
    is_agfa = 'AGFA' in vendor

    if info.get('israw', False):
        # Raw/for processing images: log transform then squared
        gmax = 2**14 - 1
        gmin = 1.0
        im_proc = np.clip(im, gmin, gmax)
        np.log(im_proc, out=im_proc)
        
        imn = (np.log(gmax) - im_proc)
        imn = np.square(imn, out=imn)
        imn = _mat2gray(imn)

    elif is_negative or is_agfa:
        # Negative polarity vendors: invert
        gmax = im.max()
        gmin = 0.0
        im_inv = np.clip(gmax - im, gmin, gmax)
        imn = _mat2gray(im_inv)

    else:
        # Default (GE, Hologic, etc.): clamp to 12-bit range then normalize
        gmax = 2**12 - 1
        gmin = 0.0
        im_n = np.clip(im, gmin, gmax)
        imn = _mat2gray(im_n)

    return imn, im


def _mat2gray(im):
    """Equivalent of MATLAB mat2gray: rescale to [0, 1]."""
    lo = im.min()
    hi = im.max()
    if hi == lo:
        return np.zeros_like(im, dtype=float)
    return (im - lo) / (hi - lo)
