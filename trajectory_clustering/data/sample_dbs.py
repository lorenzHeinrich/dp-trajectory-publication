import os
import pandas as pd

from trajectory_clustering.data.read_db import merge_t_drive_days


if __name__ == "__main__":
    os.makedirs("sample_dbs", exist_ok=True)

    t_drive = pd.read_csv(
        "t-drive-trajectories/release/taxi_log_2008_by_id/cleaned_normalized.csv"
    )

    t_drive_small = merge_t_drive_days(t_drive, 1)
    t_drive_small.to_csv("sample_dbs/t_drive_small.csv", index=False)

    t_drive_medium = merge_t_drive_days(t_drive, 3)
    t_drive_medium.to_csv("sample_dbs/t_drive_medium.csv", index=False)

    t_drive_all = merge_t_drive_days(t_drive, 7)
    t_drive_all.to_csv("sample_dbs/t_drive_all.csv", index=False)
