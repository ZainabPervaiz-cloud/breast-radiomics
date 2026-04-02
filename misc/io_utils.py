import os
import glob
import numpy as np
import pydicom

def get_dicom_info(dicom_image_path):
    """
    Retrieves information from a DICOM file (Python equivalent of getinfo.m).
    """
    info = {
        'ID': None, 'source': 'Unknown', 'israw': False, 'view': '',
        'side': '', 'tomo': '', 'istomo': False, 'ismlo': False, 'iscc': False,
        'psize': np.nan, 'H': np.nan, 'KVP': np.nan, 'target': 'NA', 'filter': 'NA'
    }
    
    if not os.path.exists(dicom_image_path):
        import warnings
        warnings.warn('The file does not exist!')
        return info
        
    try:
        ds = pydicom.dcmread(dicom_image_path)
    except Exception as e:
        return info
        
    # Series Description (Tomo, Raw)
    series_desc = ds.SeriesDescription if 'SeriesDescription' in ds else ''
    if 'Tomosynthesis' in series_desc:
        info['istomo'] = True
        info['tomo'] = 'RC' if 'Reconstruction' in series_desc else 'PR'
        info['israw'] = 'Raw' in series_desc
    
    if 'PresentationIntentType' in ds:
        info['israw'] = False if ds.PresentationIntentType == 'FOR PRESENTATION' else True
        
    if 'AccessionNumber' in ds:
        info['ID'] = ds.AccessionNumber
        
    # View Position
    view_pos = ds.ViewPosition if 'ViewPosition' in ds else ''
    protocol = ds.ProtocolName if 'ProtocolName' in ds else ''
    device_proc = ds.AcquisitionDeviceProcessingDescription if 'AcquisitionDeviceProcessingDescription' in ds else ''
    
    if view_pos == 'MLO' or 'MLO' in series_desc or 'MLO' in protocol or 'MLO' in device_proc:
        info['view'] = 'MLO'
    elif view_pos == 'CC' or 'CC' in series_desc or 'CC' in protocol or 'CC' in device_proc:
        info['view'] = 'CC'
    elif view_pos:
        info['view'] = view_pos.replace(' ', '')
        
    info['ismlo'] = (info['view'] == 'MLO')
    info['iscc'] = (info['view'] == 'CC')
    
    # Side (Laterality)
    if 'Laterality' in ds and ds.Laterality:
        info['side'] = ds.Laterality
    elif 'ImageLaterality' in ds and ds.ImageLaterality:
        info['side'] = ds.ImageLaterality
        
    if 'Manufacturer' in ds:
        info['source'] = ds.Manufacturer
        
    if 'BodyPartThickness' in ds:
        info['H'] = ds.BodyPartThickness
        
    if 'KVP' in ds:
        info['KVP'] = ds.KVP
        
    # Pixel Size
    if 'ImagerPixelSpacing' in ds:
        info['psize'] = ds.ImagerPixelSpacing[0]
    elif 'PixelSpacing' in ds:
        info['psize'] = ds.PixelSpacing[0]
        
    return info

def ffdm_read(impath, info):
    """
    Python equivalent of ffdmRead.m
    Reads DICOM, normalizes intensity.
    """
    ds = pydicom.dcmread(impath)
    im = ds.pixel_array.astype(float)
    
    vendor = info['source'].upper()
    is_negative = 'FUJIFILM' in vendor or 'SECTRA' in vendor or 'PHILIPS' in vendor
    is_agfa = 'AGFA' in vendor
    
    im_norm = im.copy()
    
    if info['israw']:
        gmax, gmin = (2**14)-1, 1
        im[im < gmin] = gmin
        im[im > gmax] = gmax
        im_norm = np.log(im)
        im_norm = (np.log(gmax) - im_norm)**2
        im_norm = (im_norm - im_norm.min()) / (im_norm.max() - im_norm.min() + 1e-8)
    elif is_negative or is_agfa:
        gmax, gmin = im.max(), 0
        im_tmp = gmax - im
        im_tmp[im_tmp < gmin] = gmin
        im_tmp[im_tmp > gmax] = gmax
        im_norm = (im_tmp - im_tmp.min()) / (im_tmp.max() - im_tmp.min() + 1e-8)
    else:
        gmax, gmin = (2**12)-1, 0
        im_norm[im > gmax] = gmax
        im_norm[im < gmin] = gmin
        im_norm = (im_norm - im_norm.min()) / (im_norm.max() - im_norm.min() + 1e-8)
        
    return im_norm, im

def read_patient_folder(folder_path):
    """
    Reads a patient folder. Retrieves all valid views (CC, MLO for L, R).
    Returns a dictionary structured by side and view:
    {
       'L': {'CC': {'image': ..., 'info': ...}, 'MLO': ...},
       'R': {'CC': ..., 'MLO': ...}
    }
    """
    patient_data = {'L': {}, 'R': {}}
    
    dicom_files = []
    for root, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if 'DICOMDIR' in filename.upper():
                continue
            path = os.path.join(root, filename)
            try:
                # pydicom has is_dicom to quickly verify preamble without full parse
                if pydicom.misc.is_dicom(path):
                    dicom_files.append(path)
                elif path.lower().endswith('.dcm') or path.lower().endswith('.ima'):
                    dicom_files.append(path)
            except:
                pass
        
    for path in dicom_files:
        info = get_dicom_info(path)
        side = info.get('side', '')
        view = info.get('view', '')
        
        if side in ['L', 'R'] and view in ['CC', 'MLO']:
            im_norm, im_raw = ffdm_read(path, info)
            patient_data[side][view] = {
                'im_norm': im_norm,
                'im_raw': im_raw,
                'info': info,
                'path': path
            }
            
    return patient_data
