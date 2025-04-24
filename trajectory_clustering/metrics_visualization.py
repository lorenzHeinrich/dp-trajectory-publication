import folium
import numpy as np

from trajectory_clustering.data.transform import lonlat_to_xy
from trajectory_clustering.metrics import (
    definitely_always_inside,
    possibly_sometimes_inside,
)

# Define the center of Karlsruhe for the map
kit_center = [49.01188, 8.41510]


def add_trajectory(m, trajectory, uncertainties):
    for idx, point in enumerate(trajectory, start=1):
        folium.Marker(
            location=point,
            popup=f"Original Point {idx}",
        ).add_to(m)

    for i, point in enumerate(trajectory):
        folium.Circle(
            location=point,
            radius=uncertainties[i],
            color="blue",
            fill=True,
            fill_opacity=0.2,
            popup="Uncertainty Region",
        ).add_to(m)

    folium.PolyLine(
        locations=trajectory,
        color="blue",
        weight=2.5,
        opacity=0.8,
    ).add_to(m)


def kit(trajectory, uncertainties):
    m = folium.Map(location=kit_center, zoom_start=17)
    add_trajectory(m, trajectory, uncertainties)
    return m


# def meters_to_degrees(meters):
#     return meters / 111320  # Approximate conversion at the equator


def add_query_region(m, center, radius, containing):
    folium.Circle(
        location=center,
        radius=radius,
        color="green" if containing else "red",
        fill=True,
        fill_opacity=0.1,
        popup="Query Region",
    ).add_to(m)
    folium.Marker(
        location=center,
        popup="Query Center",
        icon=folium.Icon(color="green" if containing else "red"),
    ).add_to(m)


# lat, lon
trajectory = [
    [49.013996, 8.419688],  # Informatik Bau
    [49.011199, 8.418734],  # InformatiKOM
    [49.011998, 8.416847],  # Mensa
    [49.012797, 8.412196],  # LZ
]

uncertainties = [100, 50, 40, 120]

D = np.array([[(lonlat_to_xy(lon, lat, "Karlsruhe")) for lat, lon in trajectory]])
U = np.array([uncertainties])
print(D)

print(D.shape)
psi_map = kit(trajectory, uncertainties)

query_center = [49.011316, 8.416873]
query_center_m = lonlat_to_xy(query_center[1], query_center[0], "Karlsruhe")
query_radius = 100

is_inside = possibly_sometimes_inside(D, (query_center_m, query_radius), (0, 3), U)
add_query_region(psi_map, query_center, query_radius, is_inside == 1)

query_center = [49.010583, 8.412597]
query_center_m = lonlat_to_xy(query_center[1], query_center[0], "Karlsruhe")
query_radius = 120

is_inside = possibly_sometimes_inside(D, (query_center_m, query_radius), (0, 3), U)

add_query_region(psi_map, query_center, query_radius, is_inside == 1)

# Display the map
psi_map.save("maps/psi_map.html")


dai_map = kit(trajectory, uncertainties)

query_center = [49.012758, 8.415633]
query_center_m = lonlat_to_xy(query_center[1], query_center[0], "Karlsruhe")
query_radius = 450
is_inside = definitely_always_inside(D, (query_center_m, query_radius), (0, 3), U)
add_query_region(dai_map, query_center, query_radius, is_inside == 1)

query_radius = 350
is_inside = definitely_always_inside(D, (query_center_m, query_radius), (0, 3), U)
add_query_region(dai_map, query_center, query_radius, is_inside == 1)

dai_map.save("maps/dai_map.html")
