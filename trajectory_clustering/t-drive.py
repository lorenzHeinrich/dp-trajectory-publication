import pandas as pd
import os
import matplotlib.pyplot as plt

directory_path = "../t-drive-trajectories/release/taxi_log_2008_by_id"
files = os.listdir(directory_path)

data_frames = []
for i in range(1, len(files) + 1):
    file_path = os.path.join(directory_path, str(i) + ".txt")
    if os.path.isfile(file_path):
        df = pd.read_csv(
            file_path,
            names=["id", "datetime", "longitude", "latitude"],
            parse_dates=["datetime"],
        )
        data_frames.append(df)
all_data = pd.concat(data_frames, ignore_index=True)

taxi_123 = all_data[all_data["id"] == 123]

# Plot the number of points per datetime
taxi_123.groupby(["datetime"]).agg({"datetime": "size"}).plot()
plt.show()

# plot the datetime difference in minutes
(taxi_123["datetime"].diff() / 60).plot()
plt.ylabel("Time Difference (minutes)")
plt.show()

taxi_123[['longitude', 'latitude']].plot.scatter(x='longitude', y='latitude')
plt.show()
