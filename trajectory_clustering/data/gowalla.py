from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# 0       2010-10-19T23:55:27Z    30.2359091167   -97.7951395833  22847

df = pd.read_csv(
    "gowalla/loc-gowalla_totalCheckins.txt",
    header=None,
    names=["user", "timestamp", "latitude", "longitude", "location_id"],
    sep=None,
    engine="python",
    nrows=10000,
)

users = df["user"].unique()
users = np.random.choice(users, 4)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    sample = df[df["user"] == users[i]]
    ax.plot(sample["longitude"], sample["latitude"], linestyle="-", marker="o")
    ax.set_title(f"User {users[i]}")
plt.show()
