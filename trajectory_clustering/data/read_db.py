import pandas as pd
import numpy as np


def csv_db_to_numpy(df: pd.DataFrame):
    df = df.sort_values(["id", "timestamp"])
    len = int(df.shape[0] / df["id"].nunique())
    n = int(df.shape[0] / len)
    D = np.empty((n, 2 * len))
    for idx, (_, group) in enumerate(df.groupby("id")):
        group = group[["longitude", "latitude"]].to_numpy()
        D[idx] = group.flatten()
    return D.reshape(n, len, 2)
