"""
Python port of readBCDR.m
Read image data from BCDR dataset CSV files.
"""
import os
import pandas as pd
import numpy as np


def readBCDR(csvfile):
    """
    Read image data from BCDR dataset.

    Parameters
    ----------
    csvfile : str
        Path to *.csv file with image data.

    Returns
    -------
    dataset : pd.DataFrame
        DataFrame with data from each image in the dataset.
    """
    fname = os.path.splitext(os.path.basename(csvfile))[0]
    is_img = fname.endswith('_img')

    # Read CSV using pandas
    try:
        df = pd.read_csv(csvfile)
    except Exception as e:
        print(f"Error reading {csvfile}: {e}")
        return pd.DataFrame()

    dataset = pd.DataFrame()

    # Patient/Study/Series
    for col in ['patient_id', 'study_id', 'series']:
        if col in df.columns:
            dataset[col] = df[col]

    # Breast density
    if 'density' in df.columns:
        dataset['density'] = df['density'].astype(np.uint8)
    elif not dataset.empty:
        dataset['density'] = np.zeros(len(dataset), dtype=np.uint8)

    # Lesion age
    if 'age' in df.columns:
        dataset['age'] = df['age']
    elif not dataset.empty:
        dataset['age'] = np.nan * np.ones(len(dataset))

    # Lesion ID
    if 'lesion_id' in df.columns:
        dataset['lesion_id'] = df['lesion_id']
    elif not dataset.empty:
        dataset['lesion_id'] = [None] * len(dataset)

    # Image path
    if 'image_filename' in df.columns:
        srcpath = os.path.dirname(csvfile)
        dataset['path'] = df['image_filename'].apply(lambda x: os.path.join(srcpath, x))

    # Classification (Malignant vs Benign)
    if 'classification' in df.columns:
        dataset['class'] = (df['classification'] == 'Malign').astype(bool)
    elif not dataset.empty:
        dataset['class'] = np.zeros(len(dataset), dtype=bool)

    # Image view
    view_list = ['RCC', 'LCC', 'RMLO', 'LMLO']
    if 'image_view' in df.columns:
        # MATLAB uses 1-based indexing for view_list
        dataset['view'] = df['image_view'].apply(lambda x: view_list[int(x)-1] if 1 <= x <= 4 else None)
    elif 'image_type_name' in df.columns:
        view_code = ['RCC', 'LCC', 'RO', 'LO']
        def map_view(v):
            try:
                idx = view_code.index(v)
                return view_list[idx]
            except ValueError:
                return None
        dataset['view'] = df['image_type_name'].apply(map_view)

    # Coordinates (ROI points)
    if 'lw_x_points' in df.columns:
        def parse_points(pts):
            if pd.isna(pts): return []
            try:
                return [float(x) for x in str(pts).split()]
            except ValueError:
                return []
        dataset['xpoints'] = df['lw_x_points'].apply(parse_points)
        dataset['ypoints'] = df['lw_y_points'].apply(parse_points)
    elif not dataset.empty:
        dataset['xpoints'] = [[] for _ in range(len(dataset))]
        dataset['ypoints'] = [[] for _ in range(len(dataset))]

    return dataset
