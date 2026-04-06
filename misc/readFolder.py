"""
Python port of readFolder.m
Read metadata from Breast Cancer Measurement Challenge directory.
"""
import os
import pandas as pd
import numpy as np


def readFolder(srcdir, tflag=True, src=1):
    """
    Read metadata from Measurement Challenge directory.

    Parameters
    ----------
    srcdir : str
        Path to source directory.
    tflag : bool
        True for training set, False for testing set.
    src : int
        Source identifier (1: Australia, 2: Malaysia, 3: Norway, 4: UK, 5: USA).

    Returns
    -------
    imdata : pd.DataFrame
        DataFrame with information of each image in the dataset.
    """
    if tflag:
        csv_path = os.path.join(srcdir, 'CoreDataTraining.csv')
        imdir = os.path.join(srcdir, 'Images', 'Training')
    else:
        csv_path = os.path.join(srcdir, 'CoreDataTesting.csv')
        imdir = os.path.join(srcdir, 'Images', 'Testing')

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return pd.DataFrame()

    t = pd.read_csv(csv_path)

    if tflag:
        # Columns in MATLAB: [1 2 3 7 10 11 12 13 14 15 8]
        # ID, year1, year2, eth, BMI, side, view, isRaw, vendor, path, class
        cols = t.iloc[:, [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 7]]
        imdata = cols.copy()
        imdata['isTrain'] = True
        imdata['source'] = src
    else:
        # Columns in MATLAB: [1 2 3 6 7 8 9 10 11 12]
        cols = t.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]]
        imdata = cols.copy()
        imdata['class'] = np.nan
        imdata['isTrain'] = False
        imdata['source'] = src

    imdata.columns = ['ID', 'year1', 'year2', 'ethnicity', 'BMI', 'side', 'view', 'isRaw', 'vendor', 'path', 'class', 'isTrain', 'source']

    # --- Mappings ---
    
    # Ethnicity mapping
    elist = imdata['ethnicity'].unique()
    e_map = {}
    print('Ethnicities:')
    for e in elist:
        e_str = str(e).lower()
        if 'asian subcontinent' in e_str: val = 1
        elif 'european' in e_str: val = 2
        elif 'hispanic' in e_str: val = 3
        elif 'missing' in e_str: val = 4
        elif 'southeast asian' in e_str: val = 5
        elif 'south-east asian' in e_str: val = 5
        elif 'african' in e_str: val = 6
        elif '1:white' in e_str: val = 7
        elif '2:black' in e_str: val = 7
        elif '4:asian' in e_str: val = 7
        else: val = 4
        e_map[e] = val
        print(f"{e}: {val}")
    imdata['ethnicity'] = imdata['ethnicity'].map(e_map)

    # Vendor mapping
    mlist = imdata['vendor'].unique()
    m_map = {}
    print('\nManufacturers:')
    for m in mlist:
        m_str = str(m).lower()
        if 'agfa' in m_str: val = 1
        elif 'fujifilm corporation' in m_str: val = 2
        elif 'hologic, inc.' in m_str: val = 3
        elif 'hologic' in m_str: val = 3
        elif 'konica minolta' in m_str: val = 4
        elif 'philips digital mammography sweden ab' in m_str: val = 5
        elif 'siemens' in m_str: val = 6
        elif 'sectra imtec ab' in m_str: val = 7
        elif 'ge' in m_str: val = 8
        else: val = 0
        m_map[m] = val
        print(f"{m}: {val}")
    imdata['vendor'] = imdata['vendor'].map(m_map)

    # Diagnosis mapping (if training)
    if tflag:
        slist = imdata['class'].unique()
        s_map = {}
        print('\nDiagnosis:')
        for s in slist:
            s_str = str(s).lower()
            if 'control' in s_str: val = 0
            elif 'invasive case' in s_str or 'unknown invasiveness' in s_str: val = 1
            elif 'in-situ case' in s_str or 'in situ case' in s_str: val = 2
            else: val = -1
            s_map[s] = val
            print(f"{s}: {val}")
        imdata['class'] = imdata['class'].map(s_map)

    # --- Processing each row ---
    def process_row(row):
        # Side
        side = 'L' if str(row['side']).lower() == 'left' else 'R'
        
        # Raw flag
        is_raw = str(row['isRaw']).lower() != 'processed'
        
        # Path fix
        fid = str(row['path'])
        fid = os.path.basename(fid)
        folder = fid.split('_')[0]
        impath = os.path.join(imdir, folder, fid)
        impath = os.path.splitext(impath)[0] + '.dcm'
        
        return side, is_raw, impath

    processed = imdata.apply(process_row, axis=1, result_type='expand')
    imdata['side'] = processed[0]
    imdata['isRaw'] = processed[1]
    imdata['path'] = processed[2]

    # --- Summary display ---
    for raw_status in [True, False]:
        subset = imdata[imdata['isRaw'] == raw_status]
        label = "RAW" if raw_status else "PROCESSED"
        print(f"\n{label} images: {len(subset)} ******")
        if not subset.empty:
            age = subset['year2'] - subset['year1']
            print(f"Age:       [{int(age.min())}-{int(age.max())}] ({int(age.median())})")
            eth = subset['ethnicity']
            counts = [sum(eth == i) for i in range(1, 7)]
            print(f"Ethnicity: 1:{counts[0]}, 2:{counts[1]}, 3:{counts[2]}, 4:{counts[3]}, 5:{counts[4]}, 6:{counts[5]}")
            bmi = subset['BMI']
            print(f"BMI:       [{bmi.min():.2f}-{bmi.max():.2f}] ({bmi.mean():.2f})")
            v = subset['vendor']
            v_counts = [sum(v == i) for i in range(1, 9)]
            print(f"Vendor:    1:{v_counts[0]}, 2:{v_counts[1]}, 3:{v_counts[2]}, 4:{v_counts[3]}, 5:{v_counts[4]}, 6:{v_counts[5]}, 7:{v_counts[6]}, 8:{v_counts[7]}")
            if tflag:
                d = subset['class']
                print(f"Diagnosis: control: {sum(d==0)}, invasive: {sum(d==1)}, in situ: {sum(d==2)}")

    return imdata
