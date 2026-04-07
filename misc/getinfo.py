"""
Python port of getinfo.m
Retrieve DICOM metadata for a mammogram image.
"""
import os
import warnings
import numpy as np
import pydicom


def getinfo(dicom_image):
    """
    Retrieve information from a DICOM file.

    Parameters
    ----------
    dicom_image : str
        Path to the DICOM file.

    Returns
    -------
    info : dict
        Dictionary with the following fields:
        ID, istomo, iscc, ismlo, view, side, israw, tomo,
        source, target, filter, KVP, psize, H, cforce,
        age, exposure, dose, date
    """
    info = {
        'ID': None,
        'source': None,
        'israw': False,
        'view': None,
        'side': None,
        'tomo': None,
        'istomo': False,
        'ismlo': False,
        'iscc': False,
        'psize': np.nan,
        'H': np.nan,
        'KVP': np.nan,
        'target': 'NA',
        'filter': 'NA',
        'cforce': np.nan,
        'age': '055Y',
        'exposure': np.nan,
        'dose': np.nan,
        'date': None,
    }

    if not os.path.exists(dicom_image):
        warnings.warn('the file does not exist!')
        return info

    try:
        ds = pydicom.dcmread(dicom_image)
    except Exception:
        return info

    # ISTOMO
    series_desc = getattr(ds, 'SeriesDescription', '') or ''
    if 'Tomosynthesis' in series_desc:
        info['istomo'] = True
        info['tomo'] = 'RC' if 'Reconstruction' in series_desc else 'PR'
        info['israw'] = 'Raw' in series_desc
    else:
        info['istomo'] = False
        info['tomo'] = None

    # ISRAW (overrides tomo-based detection if PresentationIntentType present)
    pit = getattr(ds, 'PresentationIntentType', None)
    if pit is not None:
        info['israw'] = (pit != 'FOR PRESENTATION')

    # ACCESSION NUMBER
    accnum = getattr(ds, 'AccessionNumber', None)
    if accnum is not None:
        info['ID'] = accnum

    # VIEW POSITION
    view_pos = (getattr(ds, 'ViewPosition', '') or '').strip()
    protocol = (getattr(ds, 'ProtocolName', '') or '').strip()
    device_proc = (getattr(ds, 'AcquisitionDeviceProcessingDescription', '') or '').strip()

    if view_pos == 'MLO':
        info['view'] = 'MLO'
    elif view_pos == 'CC':
        info['view'] = 'CC'
    elif 'CC' in series_desc:
        info['view'] = 'CC'
    elif 'MLO' in series_desc:
        info['view'] = 'MLO'
    elif 'MLO' in protocol:
        info['view'] = 'MLO'
    elif 'CC' in protocol:
        info['view'] = 'CC'
    elif view_pos:
        info['view'] = view_pos.replace(' ', '')
    elif device_proc:
        if 'CC' in device_proc:
            info['view'] = 'CC'
        elif 'MLO' in device_proc:
            info['view'] = 'MLO'
        else:
            info['view'] = ''

    if info['view'] is None:
        info['view'] = ''

    info['ismlo'] = (info['view'].upper() == 'MLO')
    info['iscc'] = (info['view'].upper() == 'CC')

    # SIDE (Laterality)
    lat = getattr(ds, 'Laterality', None)
    img_lat = getattr(ds, 'ImageLaterality', None)
    if lat:
        info['side'] = str(lat).strip()
    elif img_lat:
        info['side'] = str(img_lat).strip()
    else:
        info['side'] = ''

    # MANUFACTURER
    mfr = getattr(ds, 'Manufacturer', None)
    info['source'] = str(mfr).strip() if mfr else 'Unknown'

    # BREAST THICKNESS
    bpt = getattr(ds, 'BodyPartThickness', None)
    info['H'] = float(bpt) if bpt is not None else np.nan

    # KVP
    kvp = getattr(ds, 'KVP', None)
    info['KVP'] = float(kvp) if kvp is not None else np.nan

    # SPATIAL RESOLUTION / PIXEL SIZE
    if hasattr(ds, 'SpatialResolution') and ds.SpatialResolution:
        info['psize'] = float(ds.SpatialResolution)
    elif hasattr(ds, 'DetectorElementPhysicalSize') and ds.DetectorElementPhysicalSize:
        v = ds.DetectorElementPhysicalSize
        info['psize'] = float(v[0]) if hasattr(v, '__len__') else float(v)
    elif hasattr(ds, 'DetectorElementSpacing') and ds.DetectorElementSpacing:
        v = ds.DetectorElementSpacing
        info['psize'] = float(v[0]) if hasattr(v, '__len__') else float(v)
    elif hasattr(ds, 'PixelSpacing') and ds.PixelSpacing:
        v = ds.PixelSpacing
        info['psize'] = float(v[0]) if hasattr(v, '__len__') else float(v)
    elif hasattr(ds, 'ImagerPixelSpacing') and ds.ImagerPixelSpacing:
        v = ds.ImagerPixelSpacing
        info['psize'] = float(v[0]) if hasattr(v, '__len__') else float(v)
    else:
        info['psize'] = np.nan

    # ANODE TARGET
    atm = getattr(ds, 'AnodeTargetMaterial', None)
    info['target'] = str(atm)[:2] if atm else 'NA'

    # FILTER MATERIAL
    fm = getattr(ds, 'FilterMaterial', None)
    info['filter'] = str(fm)[:2] if fm else 'NA'

    # STUDY DATE
    sd = getattr(ds, 'StudyDate', None)
    if sd:
        info['date'] = str(sd)

    # iscc = NOT ismlo (matches MATLAB line 163)
    info['iscc'] = not info['ismlo']

    # COMPRESSION FORCE
    cf = getattr(ds, 'CompressionForce', None)
    if cf is not None:
        info['cforce'] = float(cf)

    # PATIENT ID and NAME
    pid = getattr(ds, 'PatientID', None)
    info['patient_id'] = str(pid) if pid else 'Unknown'
    pn = getattr(ds, 'PatientName', None)
    info['name'] = str(pn) if pn else 'Unknown'

    # PATIENT AGE
    age = getattr(ds, 'PatientAge', None)
    info['age'] = str(age) if age else '055Y'

    # EXPOSURE TIME
    et = getattr(ds, 'ExposureTime', None)
    info['exposure'] = float(et) if et is not None else np.nan

    # ORGAN DOSE
    od = getattr(ds, 'OrganDose', None)
    info['dose'] = float(od) if od is not None else np.nan

    return info
