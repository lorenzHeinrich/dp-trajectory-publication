import pandas as pd

from trajectory_clustering.data.transform import lonlat_to_xy


directory_path = "t-drive-trajectories/release/taxi_log_2008_by_id"

# Load cleaned raw data
df = pd.read_csv(f"{directory_path}/cleaned.csv")

# Filter outliers based on IQR
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

# Remove short trajectories
points_per_day = df.groupby(["id", "date"]).size().reset_index(name="counts")
df = df.merge(
    points_per_day[points_per_day["counts"] >= 37][["id", "date"]],
    on=["id", "date"],
    how="inner",
)

# Convert (lat, lon) → (x, y) using UTM (EPSG:32650 for Beijing)
xs, ys = [], []
for lon, lat in zip(df["longitude"], df["latitude"]):
    x, y = lonlat_to_xy(lon, lat, "Beijing")
    xs.append(x)
    ys.append(y)

df["x"] = xs
df["y"] = ys

df = df.drop(columns=["latitude", "longitude"])

# Save transformed data
df.to_csv(f"{directory_path}/cleaned_projected.csv", index=False)
