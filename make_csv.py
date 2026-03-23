import pandas as pd
import numpy as np
import os

np.random.seed(0)
df = pd.DataFrame({
    'PD': np.random.rand(100) * 50,
    'Age': np.random.randint(40, 70, size=100),
    'Class': np.random.randint(0, 2, size=100)
})
os.makedirs('data', exist_ok=True)
df.to_csv('data/sample_data.csv', index=False)
print("Created data/sample_data.csv")
