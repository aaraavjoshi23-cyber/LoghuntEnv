import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Bwd Packet Length Max',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Fwd IAT Total', 'Bwd IAT Total', 'Fwd PSH Flags',
    'SYN Flag Count', 'RST Flag Count', 'URG Flag Count',
    'Packet Length Mean', 'Packet Length Std', 'Average Packet Size',
    'Avg Fwd Segment Size',
]
LABEL_COL = ' Label'

ATTACK_LABELS = {
    'BENIGN': 0,
    'DDoS': 1,
    'PortScan': 2,
    'FTP-Patator': 3,
    'SSH-Patator': 3,
    'Bot': 4,
    'Infiltration': 5,
}

def load_dataset(path: str, n_samples: int = 5000):
    df = pd.read_csv(path, low_memory=False)
    df = df[[LABEL_COL] + FEATURE_COLS].dropna()
    df[LABEL_COL] = df[LABEL_COL].str.strip().map(
        lambda x: ATTACK_LABELS.get(x, 0)
    )
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    # balanced sample
    benign = df[df[LABEL_COL] == 0].sample(min(n_samples // 2, len(df[df[LABEL_COL] == 0])))
    attacks = df[df[LABEL_COL] != 0].sample(min(n_samples // 2, len(df[df[LABEL_COL] != 0])))
    combined = pd.concat([benign, attacks]).sample(frac=1).reset_index(drop=True)

    return combined[FEATURE_COLS].values, combined[LABEL_COL].values
