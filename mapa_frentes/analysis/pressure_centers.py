"""Deteccion de centros de alta (H) y baja (L) presion.

Incluye clasificacion primario/secundario estilo AEMET:
- Primario (B/A): centros profundos, aislados
- Secundario (b/a): centros menos profundos cerca de un primario

Filtros:
- Altas (H): solo si presion >= high_min_pressure (default 1012 hPa)
- Bajas (L): solo si presion <= low_max_pressure (default 1020 hPa)
- Profundidad minima: diferencia con presion media del entorno >= min_depth_hpa
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from mapa_frentes.config import AppConfig
from mapa_frentes.utils.geo import haversine


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
    depth: float = 0.0     # profundidad: diferencia con presion media del entorno


def detect_pressure_centers(
    msl_smooth: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    cfg: AppConfig,
) -> list[PressureCenter]:
    """Detecta centros de alta y baja presion con jerarquia primario/secundario.

    Pipeline:
    1. Detectar extremos locales con min/max filter
    2. Filtrar por valor absoluto (high_min_pressure, low_max_pressure)
    3. Filtrar por profundidad minima (diferencia con entorno)
    4. Filtrar por distancia minima (greedy, mas extremo primero)
    5. Clasificar primario vs secundario
    """
    pc_cfg = cfg.pressure_centers
    size = pc_cfg.filter_size
    min_dist = pc_cfg.min_distance_deg

    centers = []

    # Detectar minimos (L)
    local_min = minimum_filter(msl_smooth, size=size)
    min_mask = (msl_smooth == local_min)
    _add_centers(
        centers, "L", min_mask, msl_smooth, lats, lons, min_dist,
        max_pressure=pc_cfg.low_max_pressure,
        min_depth=pc_cfg.min_depth_hpa,
        depth_radius=pc_cfg.depth_radius_deg,
    )

    # Detectar maximos (H)
    local_max = maximum_filter(msl_smooth, size=size)
    max_mask = (msl_smooth == local_max)
    _add_centers(
        centers, "H", max_mask, msl_smooth, lats, lons, min_dist,
        min_pressure=pc_cfg.high_min_pressure,
        min_depth=pc_cfg.min_depth_hpa,
        depth_radius=pc_cfg.depth_radius_deg,
    )

    # Clasificar primario/secundario
    _classify_primary_secondary(
        centers, msl_smooth, lats, lons,
        pc_cfg.secondary_radius_deg, pc_cfg.depth_radius_deg,
    )

    return centers


def _compute_depth(
    msl: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    center_y: int,
    center_x: int,
    radius_deg: float,
) -> float:
    """Calcula profundidad de un centro: diferencia con presion media del entorno.

    Para L: depth = mean_entorno - valor_centro (positivo si es mas baja)
    Para H: depth = valor_centro - mean_entorno (positivo si es mas alta)
    Se devuelve el valor absoluto.
    """
    center_lat = lats[center_y]
    center_lon = lons[center_x]

    # Crear mascara circular
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    dlat = np.abs(lat_grid - center_lat)
    dlon = np.abs(lon_grid - center_lon)
    # Correccion coseno para distancia en grados
    cos_lat = np.cos(np.radians(center_lat))
    dist_deg = np.sqrt(dlat**2 + (dlon * cos_lat)**2)

    # Anillo: entre 1/3 del radio y el radio completo (excluir el centro)
    inner = radius_deg / 3.0
    ring_mask = (dist_deg >= inner) & (dist_deg <= radius_deg)

    if ring_mask.sum() < 4:
        return 0.0

    mean_ring = float(np.mean(msl[ring_mask]))
    center_val = float(msl[center_y, center_x])
    return abs(mean_ring - center_val)


def _add_centers(
    centers: list,
    ctype: str,
    mask: np.ndarray,
    msl: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    min_dist: float,
    min_pressure: float | None = None,
    max_pressure: float | None = None,
    min_depth: float = 2.0,
    depth_radius: float = 5.0,
):
    """Agrega centros filtrandolos por valor, profundidad y distancia minima."""
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

        # Filtro por valor absoluto
        if ctype == "H" and min_pressure is not None and val < min_pressure:
            continue
        if ctype == "L" and max_pressure is not None and val > max_pressure:
            continue

        # Filtro por profundidad (anticiclones tienen gradientes mas suaves)
        depth = _compute_depth(msl, lats, lons, ys[idx], xs[idx], depth_radius)
        effective_min_depth = min_depth if ctype == "L" else min_depth * 0.25
        if depth < effective_min_depth:
            continue

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
                depth=depth,
            ))


def _classify_primary_secondary(
    centers: list[PressureCenter],
    msl: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    secondary_radius_deg: float,
    depth_radius_deg: float,
):
    """Clasifica centros como primarios o secundarios.

    Criterios combinados:
    1. Distancia: si hay un primario a menos de secondary_radius_deg → candidato a secundario
    2. Profundidad relativa: si la profundidad es < 50% de la del primario cercano → secundario
       aunque este lejos del radio

    El centro mas extremo de cada tipo es siempre primario.
    """
    for ctype in ("L", "H"):
        type_centers = [c for c in centers if c.type == ctype]
        if not type_centers:
            continue

        primaries = []
        for center in type_centers:
            near_primary = False
            for p in primaries:
                dist_m = haversine(center.lat, center.lon, p.lat, p.lon)
                dist_deg = dist_m / 111_000.0  # aprox metros a grados

                # Criterio 1: dentro del radio de influencia
                if dist_deg < secondary_radius_deg:
                    near_primary = True
                    break

                # Criterio 2: profundidad mucho menor que primario cercano
                # (hasta 2x el radio, para captar bajas poco profundas alejadas)
                if dist_deg < secondary_radius_deg * 2:
                    if p.depth > 0 and center.depth < p.depth * 0.4:
                        near_primary = True
                        break

            if near_primary:
                center.primary = False
            else:
                center.primary = True
                primaries.append(center)
