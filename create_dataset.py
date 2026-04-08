import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)
np.random.seed(42)
n = 5000

data = {
    ' Label': np.random.choice(['BENIGN','DDoS','PortScan','Bot'], n),
    'Flow Duration': np.random.randint(0, 100000, n),
    'Total Fwd Packets': np.random.randint(1, 100, n),
    'Total Backward Packets': np.random.randint(1, 100, n),
    'Total Length of Fwd Packets': np.random.randint(0, 5000, n),
    'Total Length of Bwd Packets': np.random.randint(0, 5000, n),
    'Fwd Packet Length Max': np.random.randint(0, 1500, n),
    'Bwd Packet Length Max': np.random.randint(0, 1500, n),
    'Flow Bytes/s': np.random.uniform(0, 1000000, n),
    'Flow Packets/s': np.random.uniform(0, 1000, n),
    'Flow IAT Mean': np.random.uniform(0, 100000, n),
    'Fwd IAT Total': np.random.uniform(0, 100000, n),
    'Bwd IAT Total': np.random.uniform(0, 100000, n),
    'Fwd PSH Flags': np.random.randint(0, 2, n),
    'SYN Flag Count': np.random.randint(0, 10, n),
    'RST Flag Count': np.random.randint(0, 5, n),
    'URG Flag Count': np.random.randint(0, 2, n),
    'Packet Length Mean': np.random.uniform(0, 1500, n),
    'Packet Length Std': np.random.uniform(0, 500, n),
    'Average Packet Size': np.random.uniform(0, 1500, n),
    'Avg Fwd Segment Size': np.random.uniform(0, 1500, n),
}

pd.DataFrame(data).to_csv('data/CICIDS2017_sample.csv', index=False)
print("Dataset created successfully!")
