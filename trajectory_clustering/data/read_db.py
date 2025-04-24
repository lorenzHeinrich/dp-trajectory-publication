from typing import Literal
import pandas as pd
import numpy as np


def csv_db_to_numpy(df: pd.DataFrame):
    df = df.sort_values(["id", "timestamp"])
    len = int(df.shape[0] / df["id"].nunique())
    n = int(df.shape[0] / len)
    D = np.empty((n, 2 * len))
    for idx, (_, group) in enumerate(df.groupby("id")):
        group = group[["x", "y"]].to_numpy()
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


def t_drive(size: Literal["small", "medium", "all"] = "small"):
    """
    Load the T-Drive dataset.
    :param size: "small", "medium", or "all"
    :return: DataFrame
    """
    t_drive_data = pd.read_csv(f"sample_dbs/t_drive_{size}.csv")
    D = csv_db_to_numpy(t_drive_data)
    bounds = (
        (t_drive_data["x"].min() - 0.01, t_drive_data["x"].max() + 0.01),
        (t_drive_data["y"].min() - 0.01, t_drive_data["y"].max() + 0.01),
    )
    return D, bounds
