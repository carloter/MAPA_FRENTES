"""Utilidades geograficas: distancias, transformaciones de coordenadas."""

import numpy as np


# Radio medio de la Tierra en metros
EARTH_RADIUS = 6.371e6


def haversine(lat1, lon1, lat2, lon2):
    """Calcula la distancia haversine entre dos puntos en metros.

    Args:
        lat1, lon1: Coordenadas del punto 1 en grados.
        lat2, lon2: Coordenadas del punto 2 en grados.

    Returns:
        Distancia en metros.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS * c


def grid_spacing(lats: np.ndarray, lons: np.ndarray):
    """Calcula el espaciado del grid en metros (dx, dy) para cada punto.

    Args:
        lats: Array 1D de latitudes en grados.
        lons: Array 1D de longitudes en grados.

    Returns:
        dx: Array 2D de espaciado zonal en metros (nlat, nlon).
        dy: Array 2D de espaciado meridional en metros (nlat, nlon).
    """
    dlat = np.abs(np.diff(lats).mean())  # grados
    dlon = np.abs(np.diff(lons).mean())

    # dy es constante (aprox)
    dy_m = dlat * np.pi / 180.0 * EARTH_RADIUS

    # dx depende de la latitud
    lat_rad = np.radians(lats)
    dx_m = dlon * np.pi / 180.0 * EARTH_RADIUS * np.cos(lat_rad)

    # Expandir a 2D
    nlat = len(lats)
    nlon = len(lons)
    dx = np.broadcast_to(dx_m[:, np.newaxis], (nlat, nlon)).copy()
    dy = np.full((nlat, nlon), dy_m)

    return dx, dy


def spherical_gradient(field: np.ndarray, lats: np.ndarray, lons: np.ndarray):
    """Calcula el gradiente de un campo en coordenadas esfericas.

    Args:
        field: Array 2D (nlat, nlon).
        lats: Array 1D de latitudes en grados.
        lons: Array 1D de longitudes en grados.

    Returns:
        grad_x: Componente zonal del gradiente (d/dx) en unidades/metro.
        grad_y: Componente meridional del gradiente (d/dy) en unidades/metro.
    """
    # Asegurar que el campo es 2D (squeeze en caso de dimensiones adicionales)
    if field.ndim > 2:
        field = np.squeeze(field)
    # Si aún tiene 3+ dimensiones, extraer el primer slice temporal
    if field.ndim > 2:
        field = field[0]
    if field.ndim != 2:
        raise ValueError(f"El campo debe ser 2D, pero tiene forma {field.shape}")

    dx, dy = grid_spacing(lats, lons)

    # Gradiente centrado con np.gradient
    # np.gradient devuelve [d/d_axis0, d/d_axis1] = [d/dlat_idx, d/dlon_idx]
    grad_lat_idx, grad_lon_idx = np.gradient(field)

    # Convertir de unidades/indice a unidades/metro
    grad_y = grad_lat_idx / dy
    grad_x = grad_lon_idx / dx

    # Si las latitudes son decrecientes (N->S), invertir el signo de grad_y
    if lats[0] > lats[-1]:
        grad_y = -grad_y

    return grad_x, grad_y
