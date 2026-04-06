"""
Python port of grayrlmatrix.m
Computes the gray-level run-length (GLRL) matrix.
"""
import numpy as np


def grayrlmatrix(im, offset, nlevels, gl, mask=None):
    """
    Computes the gray-level run-length (GLRL) matrix.

    Parameters
    ----------
    im : ndarray
        Input image.
    offset : list
        Directions (1: 0, 2: 45, 3: 90, 4: 135).
    nlevels : int
        Number of gray levels.
    gl : list
        [min, max] gray limits.
    mask : ndarray, optional
        Binary mask.

    Returns
    -------
    glrlms : list
        List of GLRL matrices for each direction.
    """
    im = np.asarray(im, dtype=float)
    if mask is None:
        mask = np.ones_like(im, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    # Scale image to [1, nlevels] (MATLAB line 80-87)
    if gl[0] == gl[1]:
        si = np.ones_like(im, dtype=int)
    else:
        slope = (nlevels - 1) / (gl[1] - gl[0])
        intercept = 1 - (slope * gl[0])
        si = np.round(slope * im + intercept).astype(int)

    si[si > nlevels] = nlevels
    si[si < 1] = 1

    glrlms = []
    for off in offset:
        glrlms.append(compute_glrlm(si, off, nlevels, mask))
    
    return glrlms


def compute_glrlm(si, offset, nl, mask):
    if offset == 1: # 0 degree
        return rle_0(si, nl, mask)
    elif offset == 2: # 45 degree
        seq, seq_mask = zigzag_scan(si, mask)
        return rle_diag(seq, nl, seq_mask)
    elif offset == 3: # 90 degree
        return rle_0(si.T, nl, mask.T)
    elif offset == 4: # 135 degree
        # flip left-right for 135
        seq, seq_mask = zigzag_scan(np.fliplr(si), np.fliplr(mask))
        return rle_diag(seq, nl, seq_mask)
    else:
        raise ValueError("Only offsets 1, 2, 3, 4 supported")


def rle_0(si, nl, mask):
    m, n = si.shape
    oneglrlm = np.zeros((nl, n))
    for i in range(m):
        row = si[i, :]
        m_row = mask[i, :]
        x = row[m_row]
        if len(x) > 0:
            # find runs
            locs = np.where(x[:-1] != x[1:])[0]
            index = np.append(locs, len(x) - 1)
            lengths = np.diff(np.append(-1, index))
            values = x[index]
            # Accumulate
            # In Python, we can use np.add.at
            # values is 1-based, we need 0-based for indexing nl
            # lengths is 1-based, index n-1
            np.add.at(oneglrlm, (values.astype(int) - 1, lengths.astype(int) - 1), 1)
    return oneglrlm


def rle_diag(seq, nl, seq_mask):
    # max possible run length
    n_max = 0
    for s in seq:
        if len(s) > n_max: n_max = len(s)
    
    oneglrlm = np.zeros((nl, n_max))
    for i in range(len(seq)):
        x = np.asarray(seq[i])
        m = np.asarray(seq_mask[i])
        x = x[m]
        if len(x) > 0:
            locs = np.where(x[:-1] != x[1:])[0]
            index = np.append(locs, len(x) - 1)
            lengths = np.diff(np.append(-1, index))
            values = x[index]
            np.add.at(oneglrlm, (values.astype(int) - 1, lengths.astype(int) - 1), 1)
    return oneglrlm


def zigzag_scan(si, mask):
    """
    Zigzag scan to handle diagonal directions (45/135).
    MATLAB logic from zigzag subfunction.
    """
    rows, cols = si.shape
    # Diagonal scan lines
    # A simple way to get all diagonals:
    # for d = 0 to (rows + cols - 2)
    # points (r, c) such that r + c = d
    seq = []
    seq_mask = []
    
    for d in range(rows + cols - 1):
        if d % 2 == 0:
            # Down-left
            r = min(d, rows - 1)
            c = d - r
            line = []
            m_line = []
            while r >= 0 and c < cols:
                line.append(si[r, c])
                m_line.append(mask[r, c])
                r -= 1
                c += 1
            seq.append(line)
            seq_mask.append(m_line)
        else:
            # Up-right
            c = min(d, cols - 1)
            r = d - c
            line = []
            m_line = []
            while c >= 0 and r < rows:
                line.append(si[r, c])
                m_line.append(mask[r, c])
                c -= 1
                r += 1
            seq.append(line)
            seq_mask.append(m_line)
            
    return seq, seq_mask
