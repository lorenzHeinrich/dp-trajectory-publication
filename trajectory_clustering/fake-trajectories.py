import random
import numpy as np
import csv
import animate
import pandas as pd

grid_size = 200
num_points = 100
num_users = 20
max_velocity = 20

def constrain_movement(x, y):
    positive_x = min(max_velocity, grid_size - x)
    negative_x = min(max_velocity, x)
    new_x = x + round(random.uniform(-negative_x, positive_x))
    positive_y = min(max_velocity, grid_size - y)
    negative_y = min(max_velocity, y)
    new_y = y + round(random.uniform(-negative_y, positive_y))
    return new_x, new_y

data = np.ndarray((num_users * num_points, 4))
for user in range(num_users):
    x = random.randint(0, grid_size)
    y = random.randint(0, grid_size)
    for point in range(num_points):
        x, y = constrain_movement(x, y)
        data[user * num_points + point] = [user, point, x, y]
data = data.astype(int)

with open("fake-trajectories.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "timestamp", "longitude", "latitude"])
    writer.writerows(data)
