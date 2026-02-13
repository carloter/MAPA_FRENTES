"""Deteccion de centros de alta (H) y baja (L) presion.

Incluye clasificacion primario/secundario estilo AEMET:
- Primario (B/A): centros profundos, aislados
- Secundario (b/a): centros menos profundos cerca de un primario
"""

from dataclasses import dataclass, field

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
    primary: bool = True   # True = primario (B/A), False = secundario (b/a)
    name: str = ""         # nombre de borrasca (ej: "Nils"), vacio = sin nombre
    id: str = ""           # identificador unico (ej: "L_000")


def detect_pressure_centers(
    msl_smooth: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    cfg: AppConfig,
) -> list[PressureCenter]:
    """Detecta centros de alta y baja presion con jerarquia primario/secundario.

    Pipeline:
    1. Detectar extremos locales con min/max filter
    2. Filtrar por distancia minima (greedy, mas extremo primero)
    3. Clasificar primario vs secundario
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

    # Clasificar primario/secundario
    _classify_primary_secondary(centers, pc_cfg.secondary_radius_deg)

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
            idx_type = sum(1 for c in centers if c.type == ctype)
            centers.append(PressureCenter(
                type=ctype, lat=lat, lon=lon, value=val,
                id=f"{ctype}_{idx_type:03d}",
            ))


def _classify_primary_secondary(
    centers: list[PressureCenter],
    secondary_radius_deg: float,
):
    """Clasifica centros como primarios o secundarios.

    Para cada tipo (H/L), el centro mas extremo es siempre primario.
    Los siguientes son primarios si no hay otro primario dentro del
    radio de influencia. Si hay un primario cerca, es secundario.
    """
    for ctype in ("L", "H"):
        type_centers = [c for c in centers if c.type == ctype]
        if not type_centers:
            continue

        # Ya estan ordenados por intensidad (mas extremo primero)
        # El primero siempre es primario
        primaries = []
        for center in type_centers:
            near_primary = False
            for p in primaries:
                dist = np.sqrt((center.lat - p.lat)**2 + (center.lon - p.lon)**2)
                if dist < secondary_radius_deg:
                    near_primary = True
                    break
            if near_primary:
                center.primary = False
            else:
                center.primary = True
                primaries.append(center)
