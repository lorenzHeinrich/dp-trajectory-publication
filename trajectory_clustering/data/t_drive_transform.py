from datetime import datetime, timedelta
import pandas as pd
import os

from trajectory_clustering.data.read_db import merge_t_drive_days
from trajectory_clustering.data.transform import lonlat_to_xy


def round_to_nearest_10_minutes(dt: datetime):
    total_minutes = dt.hour * 60 + dt.minute
    rounded_minutes = (total_minutes + 5) // 10 * 10
    if rounded_minutes >= 24 * 60:
        rounded_minutes -= 24 * 60
        dt += timedelta(days=1)
    new_hour = rounded_minutes // 60
    new_minute = rounded_minutes % 60
    return dt.replace(hour=new_hour, minute=new_minute, second=0, microsecond=0)


directory_path = "t-drive-trajectories/release/taxi_log_2008_by_id"
out_dir = "sample_dbs"

files = os.listdir(directory_path)
data_frames = []
print(f"Loading {len(files)} files...")
for i in range(1, len(files) + 1):
    file_path = os.path.join(directory_path, str(i) + ".txt")
    if os.path.isfile(file_path):
        # 1,2008-02-02 15:36:08,116.51172,39.92123
        df = pd.read_csv(
            file_path,
            names=["id", "timestamp", "longitude", "latitude"],
            parse_dates=["timestamp"],
            date_format="%Y-%m-%d %H:%M:%S",
        )
        if not df.empty:
            data_frames.append(df)
all_data = pd.concat(data_frames, ignore_index=True)

print("Removing outliers...")
# Filter outliers based on IQR
lat_iqr = all_data["latitude"].quantile(0.75) - all_data["latitude"].quantile(0.25)
long_iqr = all_data["longitude"].quantile(0.75) - all_data["longitude"].quantile(0.25)

lat_upr, lat_lwr = (
    all_data["latitude"].quantile(0.75) + 1.5 * lat_iqr,
    all_data["latitude"].quantile(0.25) - 1.5 * lat_iqr,
)
long_upr, long_lwr = (
    all_data["longitude"].quantile(0.75) + 1.5 * long_iqr,
    all_data["longitude"].quantile(0.25) - 1.5 * long_iqr,
)

no_outliers = all_data[
    (all_data["latitude"] >= lat_lwr)
    & (all_data["latitude"] <= lat_upr)
    & (all_data["longitude"] >= long_lwr)
    & (all_data["longitude"] <= long_upr)
]
print(f"Removed {len(all_data) - len(no_outliers)} outliers")
all_data = no_outliers

print("Filtering by time interval...")
all_data.loc[:, "timestamp"] = all_data["timestamp"].apply(round_to_nearest_10_minutes)

all_data = all_data[
    (all_data["timestamp"].dt.time >= datetime.strptime("08:30", "%H:%M").time())
    & (all_data["timestamp"].dt.time <= datetime.strptime("14:30", "%H:%M").time())
]

all_data.drop_duplicates(subset=["id", "timestamp"], inplace=True)

all_data = all_data.copy()
all_data.loc[:, "date"] = all_data["timestamp"].dt.date
by_day = all_data.groupby(["id", "date"]).size().reset_index(name="count")
all_data = pd.merge(all_data, by_day, on=["id", "date"], how="left")
all_data = all_data[all_data["count"] == 37].copy()
all_data.drop(columns=["count"], inplace=True)


# Remove short trajectories
points_per_day = all_data.groupby(["id", "date"]).size().reset_index(name="counts")
all_data = all_data.merge(
    points_per_day[points_per_day["counts"] >= 37][["id", "date"]],
    on=["id", "date"],
    how="inner",
)


print(f"Converting {len(all_data)} points to (x, y) coordinates...")
# Convert (lat, lon) → (x, y) using UTM (EPSG:32650 for Beijing)
xs, ys = [], []
for lon, lat in zip(all_data["longitude"], all_data["latitude"]):
    x, y = lonlat_to_xy(lon, lat, "Beijing")
    xs.append(x)
    ys.append(y)

all_data["x"] = xs
all_data["y"] = ys

t_drive = all_data.drop(columns=["latitude", "longitude"])


os.makedirs("sample_dbs", exist_ok=True)

t_drive_small = merge_t_drive_days(t_drive, 1)
print(f"Saving {len(t_drive_small) / 37} trajectories to {out_dir}/t_drive_small.csv")
t_drive_small.to_csv(f"{out_dir}/t_drive_small.csv", index=False)

t_drive_medium = merge_t_drive_days(t_drive, 3)
print(
    f"Saving {len(t_drive_medium) / 37 } trajectories to {out_dir}/t_drive_medium.csv"
)
t_drive_medium.to_csv(f"{out_dir}/t_drive_medium.csv", index=False)

t_drive_all = merge_t_drive_days(t_drive, 7)
print(f"Saving {len(t_drive_all) / 37} trajectories to {out_dir}/t_drive_all.csv")
t_drive_all.to_csv(f"{out_dir}/t_drive_all.csv", index=False)
print("Done!")
