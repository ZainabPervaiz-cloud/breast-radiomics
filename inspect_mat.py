import scipy.io
import pandas as pd
import numpy as np

mat = scipy.io.loadmat('data/sample_data.mat')
dataset = mat['dataset']

print("Shape:", dataset.shape)
print("Type:", type(dataset))
print("Dtype names:", dataset.dtype.names)

if dataset.dtype.names:
    print("Extracting fields...")
    df_dict = {}
    for name in dataset.dtype.names:
        val = dataset[name][0, 0]
        print(f"Field {name} shape:", val.shape)
        # Handle 2D arrays like (N, 1) to 1D
        df_dict[name] = val.flatten()
        
    df = pd.DataFrame(df_dict)
    print("DataFrame shape:", df.shape)
    df.to_csv('data/sample_data.csv', index=False)
    print("Successfully exported data/sample_data.csv")
else:
    print("Could not find dtype names.")
