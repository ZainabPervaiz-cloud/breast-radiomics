"""
Python port of getDicomFiles.m
Recursively search a folder for valid DICOM files.
"""
import os
import pydicom


def getDicomFiles(folder_path):
    """
    Recursively search folder for DICOM files.

    Parameters
    ----------
    folder_path : str
        Root folder to search.

    Returns
    -------
    file_list : list of str
        Absolute paths to all DICOM files found.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f'Folder does not exist: {folder_path}')

    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.upper() == 'DICOMDIR':
                continue
            full_path = os.path.join(root, fname)
            try:
                if pydicom.misc.is_dicom(full_path):
                    file_list.append(full_path)
            except Exception:
                # Fallback: accept .dcm / .ima by extension
                ext = os.path.splitext(fname)[1].lower()
                if ext in ('.dcm', '.ima'):
                    file_list.append(full_path)
    return file_list
