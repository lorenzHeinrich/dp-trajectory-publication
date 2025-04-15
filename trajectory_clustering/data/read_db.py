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


def merge_t_drive_days(t_drive_data: pd.DataFrame, n_days: int):
    """
    Merge the T-Drive data for each day into a single DataFrame.
    """
    days = t_drive_data["date"].unique()
    merged_data = []
    for day in days[:n_days]:
        day_data = t_drive_data.loc[t_drive_data["date"] == day].copy()

        day_data["id"] = day_data["id"].astype(str) + "_" + day_data["date"].astype(str)

        merged_data.append(day_data)

    return pd.concat(merged_data, ignore_index=True)
