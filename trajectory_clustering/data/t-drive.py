# %%
from datetime import datetime, timedelta
import pandas as pd
import os


# %%
directory_path = "t-drive-trajectories/release/taxi_log_2008_by_id"


# %%
files = os.listdir(directory_path)
data_frames = []
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


# %%
all_data = pd.concat(data_frames, ignore_index=True)


# %%
def round_to_nearest_10_minutes(dt: datetime):
    total_minutes = dt.hour * 60 + dt.minute
    rounded_minutes = (total_minutes + 5) // 10 * 10
    if rounded_minutes >= 24 * 60:
        rounded_minutes -= 24 * 60
        dt += timedelta(days=1)
    new_hour = rounded_minutes // 60
    new_minute = rounded_minutes % 60
    return dt.replace(hour=new_hour, minute=new_minute, second=0, microsecond=0)


# %%
# round the timestamp field to the nearest 10 minutes
all_data.loc[:, "timestamp"] = all_data["timestamp"].apply(round_to_nearest_10_minutes)


# %%
# filter out the data points outside of the 8:30 - 14:30 time range
all_data = all_data[
    (all_data["timestamp"].dt.time >= datetime.strptime("08:30", "%H:%M").time())
    & (all_data["timestamp"].dt.time <= datetime.strptime("14:30", "%H:%M").time())
]


# %%
# remove duplicate entries with the same id and timestamp
all_data.drop_duplicates(subset=["id", "timestamp"], inplace=True)


# %%
# keep only trajectories with 37 recordings per day, e.g. continuous recordings every 10 minutes
all_data["date"] = all_data["timestamp"].dt.date
by_day = all_data.groupby(["id", "date"]).size().reset_index(name="count")
all_data = pd.merge(all_data, by_day, on=["id", "date"], how="left")
all_data = all_data[all_data["count"] == 37]
all_data.drop(columns=["count"], inplace=True)

# %%
all_data.to_csv(f"{directory_path}/cleaned.csv", index=False)
