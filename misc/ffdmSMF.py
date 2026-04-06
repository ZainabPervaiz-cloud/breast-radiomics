"""
Python port of ffdmSMF.m
Standardize raw FFDM mammography to SMF (van Engeland 2006).
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.morphology import binary_erosion, disk


def ffdmSMF(im, info, mask):
    """
    Standardize raw FFDM mammography image to SMF (Standardized Mammography Format).

    Parameters
    ----------
    im : ndarray
        Raw mammogram image (integer values).
    info : dict
        Metadata as returned by getinfo().
    mask : ndarray
        Binary breast segmentation mask.

    Returns
    -------
    imn : ndarray
        Standardized image (volumetric density proxy).
    info : dict
        Unchanged info dict (returned for compatibility).
    """
    gap = 10  # mm gap from breast boundary

    im = np.asarray(im, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if mask.shape != im.shape:
        from skimage.transform import resize
        mask = resize(mask, im.shape, order=0, preserve_range=True).astype(bool)

    target = (info.get('target') or 'NA').upper()[:2]
    filt = (info.get('filter') or 'NA').upper()[:2]

    # Table I from van Engeland 2006: Diffs of attenuation coefficients
    h_vals = np.arange(20, 91, 10, dtype=float)   # 20:10:90 → 8 values
    kvp_vals = np.arange(24, 33, dtype=float)      # 24:32 → 9 values

    if target == 'RH' and filt == 'RH':
        Du = np.array([
            [.429, .393, .369, .351, .337, .326, .318, .310],
            [.406, .373, .351, .335, .322, .313, .304, .298],
            [.388, .358, .338, .323, .312, .302, .295, .288],
            [.374, .346, .327, .313, .302, .293, .285, .278],
            [.363, .337, .319, .305, .294, .285, .277, .269],
            [.353, .328, .310, .297, .286, .276, .267, .259],
            [.344, .320, .303, .289, .277, .267, .258, .249],
            [.336, .312, .295, .281, .268, .257, .247, .238],
            [.328, .304, .286, .272, .259, .247, .236, .226],
        ])
    elif target == 'MO' and filt == 'RH':
        Du = np.array([
            [.448, .415, .392, .373, .358, .346, .335, .326],
            [.432, .404, .379, .361, .345, .333, .322, .313],
            [.422, .392, .370, .352, .336, .323, .312, .303],
            [.413, .384, .361, .342, .327, .313, .301, .291],
            [.407, .377, .354, .335, .318, .303, .291, .280],
            [.399, .369, .345, .325, .307, .291, .277, .265],
            [.392, .361, .336, .314, .295, .278, .263, .249],
            [.384, .352, .326, .302, .282, .263, .247, .233],
            [.377, .344, .316, .291, .269, .250, .233, .219],
        ])
    elif target == 'MO' and filt == 'MO':
        Du = np.array([
            [.513, .477, .452, .433, .417, .403, .390, .379],
            [.497, .761, .435, .414, .396, .379, .363, .348],
            [.484, .448, .421, .398, .378, .358, .340, .324],
            [.468, .432, .403, .377, .354, .333, .313, .295],
            [.456, .419, .388, .360, .335, .312, .291, .273],
            [.442, .403, .370, .340, .313, .289, .267, .249],
            [.429, .387, .352, .320, .292, .267, .246, .228],
            [.414, .371, .334, .301, .272, .247, .227, .210],
            [.402, .357, .319, .285, .256, .232, .212, .196],
        ])
    else:
        raise ValueError(f'Data not available for target: {target} and filter: {filt}')

    # Interpolate udiff at the clinical KVP and H
    KVP_ref = float(np.clip(info.get('KVP', 28), kvp_vals.min(), kvp_vals.max()))
    h_ref = float(np.clip(info.get('H', 50), h_vals.min(), h_vals.max()))

    # scipy RegularGridInterpolator: rows=KVP, cols=H
    interp = RegularGridInterpolator((kvp_vals, h_vals), Du, method='linear', bounds_error=False, fill_value=None)
    udiff = float(interp([[KVP_ref, h_ref]])[0])

    # Erode mask to get inner breast (exclude border + gap)
    psize = info.get('psize', 0.1) or 0.1
    radius_px = max(1, round(0.5 * gap / psize))
    mask_inner = mask.copy()
    mask_inner[0, :] = False
    mask_inner[-1, :] = False
    mask_inner[:, 0] = False
    mask_inner[:, -1] = False
    mask_inner = binary_erosion(mask_inner, disk(radius_px))

    # Reference pixel value (fatty tissue = brightest in inner mask)
    g_fat = float(im[mask_inner].max())

    # Compute standardized image
    imn = -np.log(im / g_fat) / udiff

    return imn, info
