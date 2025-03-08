from matplotlib import pyplot as plt
import pandas as pd

directory_path = "t-drive-trajectories/release/taxi_log_2008_by_id"

df = pd.read_csv(f"{directory_path}/cleaned.csv")

lat_iqr = df["latitude"].quantile(0.75) - df["latitude"].quantile(0.25)
long_iqr = df["longitude"].quantile(0.75) - df["longitude"].quantile(0.25)
lat_upr, lat_lwr = (
    df["latitude"].quantile(0.75) + 1.5 * lat_iqr,
    df["latitude"].quantile(0.25) - 1.5 * lat_iqr,
)
long_upr, long_lwr = (
    df["longitude"].quantile(0.75) + 1.5 * long_iqr,
    df["longitude"].quantile(0.25) - 1.5 * long_iqr,
)
df = df[
    (df["latitude"] >= lat_lwr)
    & (df["latitude"] <= lat_upr)
    & (df["longitude"] >= long_lwr)
    & (df["longitude"] <= long_upr)
]

# remove trajectories with less then 37 points per date
points_per_day = df.groupby(["id", "date"]).size().reset_index(name="counts")
df = df.merge(
    points_per_day[points_per_day["counts"] >= 37][["id", "date"]],
    on=["id", "date"],
    how="inner",
)

# normalize the data
min_long, max_long = df["longitude"].min(), df["longitude"].max()
min_lat, max_lat = df["latitude"].min(), df["latitude"].max()
df["longitude"] = (df["longitude"] - min_long) / (max_long - min_long) * 100
df["latitude"] = (df["latitude"] - min_lat) / (max_lat - min_lat) * 100

df.to_csv(f"{directory_path}/cleaned_normalized.csv", index=False)


# ts = ["12:00:00", "12:10:00", "12:20:00", "12:30:00"]

# fig, axs = plt.subplots(2, 2)

# for i, ax in enumerate(axs.flat):
#     sample = df[df["timestamp"] == f"2008-02-03 {ts[i]}"]
#     ax.scatter(sample["longitude"], sample["latitude"])
#     ax.set_title(ts[i])
# plt.show()
