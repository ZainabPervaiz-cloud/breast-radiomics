"""
Python port of fixpath.m
Fix image path - replace prefix and extension.
"""
import os


def fixpath(srcpath, prefix, no, ext='.dcm'):
    """
    Fix image path.

    Parameters
    ----------
    srcpath : str or list
        Input path (or list of paths).
    prefix : str
        New prefix directory.
    no : int
        Index (1-based, matches MATLAB) from which srcpath is kept.
    ext : str
        File extension. Default '.dcm'.

    Returns
    -------
    fpath : str or list
        Fixed path(s).
    """
    if isinstance(srcpath, (list, tuple)):
        return [fixpath(p, prefix, no, ext) for p in srcpath]

    # Normalize separators (MATLAB: strrep to filesep)
    sp = srcpath.replace('/', os.sep).replace('\\', os.sep)
    # no is 1-based (MATLAB), convert to 0-based Python slice
    tail = sp[no - 1:]
    fpath = os.path.join(prefix, tail)
    srcdir = os.path.dirname(fpath)
    fname = os.path.splitext(os.path.basename(fpath))[0]
    fpath = os.path.join(srcdir, fname + ext)
    return fpath
