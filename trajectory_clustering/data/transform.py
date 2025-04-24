from typing import Literal
from pyproj import Transformer

EPSG_CODES = {
    "World": "EPSG:4326",  # WGS84
    "Beijing": "EPSG:32650",  # UTM zone 50N (covers Beijing)
    "Karlsruhe": "EPSG:25832",  # UTM zone 32N (covers Karlsruhe)
}

EPSG_WGS84 = EPSG_CODES["World"]


def ensure_lonlat(lon, lat):
    """
    Helper to enforce argument order and provide error checking.
    """
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError(
            f"Invalid input order: expected (lon, lat), got ({lon}, {lat})"
        )
    return lon, lat


def lonlat_to_xy(
    lon: float, lat: float, zone: Literal["Beijing", "Karlsruhe"]
) -> tuple[float, float]:
    """
    Convert geographic coordinates (lon, lat) to Cartesian (x, y) in meters using the given zone.
    """
    lon, lat = ensure_lonlat(lon, lat)
    epsg_target = EPSG_CODES.get(zone)
    if epsg_target is None:
        raise ValueError(f"Unsupported EPSG zone: {zone}")
    transformer = Transformer.from_crs(EPSG_WGS84, epsg_target, always_xy=True)
    return transformer.transform(lon, lat)


def xy_to_lonlat(
    x: float, y: float, zone: Literal["Beijing", "Karlsruhe"]
) -> tuple[float, float]:
    """
    Convert Cartesian coordinates (x, y) in meters to geographic (lon, lat).
    """
    epsg_target = EPSG_CODES.get(zone)
    if epsg_target is None:
        raise ValueError(f"Unsupported EPSG zone: {zone}")
    transformer = Transformer.from_crs(epsg_target, EPSG_WGS84, always_xy=True)
    return transformer.transform(x, y)
