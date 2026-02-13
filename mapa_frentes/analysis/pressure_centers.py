"""Deteccion de centros de alta (H) y baja (L) presion."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from mapa_frentes.config import AppConfig


@dataclass
class PressureCenter:
    """Representa un centro de presion H o L."""
    type: str          # "H" o "L"
    lat: float
    lon: float
    value: float       # presion en hPa


def detect_pressure_centers(
    msl_smooth: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    cfg: AppConfig,
) -> list[PressureCenter]:
    """Detecta centros de alta y baja presion.

    Usa minimum_filter y maximum_filter de scipy para encontrar
    extremos locales en el campo de MSLP suavizado.

    Args:
        msl_smooth: Array 2D de MSLP suavizada en hPa.
        lats: Array 1D de latitudes.
        lons: Array 1D de longitudes.
        cfg: Configuracion.

    Returns:
        Lista de PressureCenter.
    """
    pc_cfg = cfg.pressure_centers
    size = pc_cfg.filter_size
    min_dist = pc_cfg.min_distance_deg

    centers = []

    # Detectar minimos (L)
    local_min = minimum_filter(msl_smooth, size=size)
    min_mask = (msl_smooth == local_min)
    _add_centers(centers, "L", min_mask, msl_smooth, lats, lons, min_dist)

    # Detectar maximos (H)
    local_max = maximum_filter(msl_smooth, size=size)
    max_mask = (msl_smooth == local_max)
    _add_centers(centers, "H", max_mask, msl_smooth, lats, lons, min_dist)

    return centers


def _add_centers(
    centers: list,
    ctype: str,
    mask: np.ndarray,
    msl: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    min_dist: float,
):
    """Agrega centros filtrandolos por distancia minima."""
    ys, xs = np.where(mask)
    # Ordenar por intensidad (mas extremo primero)
    if ctype == "L":
        order = np.argsort(msl[ys, xs])
    else:
        order = np.argsort(-msl[ys, xs])

    for idx in order:
        lat = float(lats[ys[idx]])
        lon = float(lons[xs[idx]])
        val = float(msl[ys[idx], xs[idx]])

        # Verificar distancia minima a centros ya aceptados del mismo tipo
        too_close = False
        for c in centers:
            if c.type == ctype:
                dlat = abs(c.lat - lat)
                dlon = abs(c.lon - lon)
                if dlat < min_dist and dlon < min_dist:
                    too_close = True
                    break
        if not too_close:
            centers.append(PressureCenter(type=ctype, lat=lat, lon=lon, value=val))
