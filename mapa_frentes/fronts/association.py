"""Asociacion de frentes TFP a centros de baja presion.

Asigna cada frente detectado por TFP al centro L mas cercano
como metadata (center_id, association_end), sin modificar la geometria.

Ofrece opcionalmente smooth_extend_to_center() para extender un frente
hasta un centro con una curva suave tangente (Bezier cubica).
"""

from __future__ import annotations

import logging
import numpy as np

from mapa_frentes.analysis.pressure_centers import PressureCenter
from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


# -----------------------------------------------------------------------------
# Distancias robustas (haversine) + helpers
# -----------------------------------------------------------------------------

def _wrap_lon(lon: float) -> float:
    """Normaliza longitudes a [-180, 180)."""
    x = (lon + 180.0) % 360.0 - 180.0
    # evita -180 exacto si quieres consistencia
    return float(x)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Distancia haversine (km). Soporta arrays.
    lat/lon en grados.
    """
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # wrap del delta lon a [-pi, pi]
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def _deg_to_km_approx(deg: float, lat_ref: float = 45.0) -> float:
    """
    Convierte "grados" a km de forma aproximada usando lat_ref para lon.
    Útil para convertir umbrales del config (en grados) a km.
    """
    # 1 deg lat ~ 111 km
    km_lat = 111.0 * deg
    km_lon = 111.0 * np.cos(np.deg2rad(lat_ref)) * deg
    # usa algo intermedio
    return float(0.5 * (km_lat + km_lon))


def _front_min_distance_km(front: Front, center: PressureCenter) -> float:
    """Distancia mínima (km) desde cualquier punto del frente al centro."""
    lats = front.lats
    lons = np.array([_wrap_lon(x) for x in front.lons], dtype=float)
    c_lat = center.lat
    c_lon = _wrap_lon(center.lon)

    d = _haversine_km(lats, lons, c_lat, c_lon)
    return float(np.nanmin(d)) if d.size else float("inf")


def _front_end_distances_km(front: Front, center: PressureCenter) -> tuple[float, float]:
    """Distancias (km) centro->extremo start y centro->extremo end."""
    c_lat = center.lat
    c_lon = _wrap_lon(center.lon)
    d_start = float(_haversine_km(front.lats[0], _wrap_lon(front.lons[0]), c_lat, c_lon))
    d_end = float(_haversine_km(front.lats[-1], _wrap_lon(front.lons[-1]), c_lat, c_lon))
    return d_start, d_end


# -----------------------------------------------------------------------------
# Asociacion
# -----------------------------------------------------------------------------

def find_nearest_center_for_front(
    front: Front,
    lows: list[PressureCenter],
    *,
    strategy: str = "min_along_front",   # "ends" o "min_along_front"
) -> tuple[PressureCenter | None, str | None, float]:
    """Encuentra el centro L mas cercano y el extremo a asociar.

    strategy:
      - "ends": solo compara distancias a start/end (legacy)
      - "min_along_front": elige el centro por distancia mínima a cualquier punto;
        el extremo asociado se decide por cuál extremo queda más cerca del centro.

    Returns:
        (center, which_end, distance_km)
    """
    best_dist = float("inf")
    best_center = None
    best_end = None

    for center in lows:
        if strategy == "ends":
            d_start, d_end = _front_end_distances_km(front, center)
            if d_start < d_end and d_start < best_dist:
                best_dist, best_center, best_end = d_start, center, "start"
            elif d_end <= d_start and d_end < best_dist:
                best_dist, best_center, best_end = d_end, center, "end"
        else:
            # robusto: decide centro por mínima distancia a lo largo del frente
            d_min = _front_min_distance_km(front, center)
            if d_min < best_dist:
                # qué extremo es "mejor" para extender
                d_start, d_end = _front_end_distances_km(front, center)
                which = "start" if d_start < d_end else "end"
                best_dist, best_center, best_end = d_min, center, which

    return best_center, best_end, best_dist


def associate_fronts_to_centers(
    collection: FrontCollection,
    centers: list[PressureCenter],
    cfg: AppConfig,
) -> FrontCollection:
    """Asocia frentes existentes al centro L mas cercano (solo metadata o extend opcional).

    Para cada frente:
      1) Busca mejor centro L con estrategia robusta
      2) Si distancia <= umbral, asigna center_id y association_end
      3) Opcional: extender geométricamente hacia el centro (Bezier + spline)
    """
    lows = [c for c in centers if c.type == "L"]
    if not lows:
        return collection

    # Umbral en km (partimos del config en grados, para no romper tu YAML)
    max_dist_deg = float(cfg.center_fronts.max_association_distance_deg)
    lat_ref = float(getattr(cfg.plotting.projection_params, "central_latitude", 45.0)) if hasattr(cfg, "plotting") else 45.0
    max_dist_km = _deg_to_km_approx(max_dist_deg, lat_ref=lat_ref)

    # Nuevos flags (opcionales)
    strategy = getattr(cfg.center_fronts, "association_strategy", "min_along_front")
    do_extend = bool(getattr(cfg.center_fronts, "extend_to_center", False))
    extend_only_if_close_km = float(getattr(cfg.center_fronts, "extend_max_distance_km", max_dist_km))

    associated = 0
    extended = 0

    for front in collection:
        if front.front_type == FrontType.INSTABILITY_LINE:
            continue
        if front.center_id:
            continue

        best_center, best_end, best_dist_km = find_nearest_center_for_front(
            front, lows, strategy=strategy
        )

        if best_center is None or best_end is None:
            continue

        if best_dist_km <= max_dist_km:
            front.center_id = best_center.id
            front.association_end = best_end
            associated += 1

            # Extensión geométrica opcional (esto ayuda a que “nazca” del centro como en análisis manual)
            if do_extend and best_dist_km <= extend_only_if_close_km:
                try:
                    smooth_extend_to_center(front, best_center, best_end, cfg)
                    extended += 1
                except Exception as e:
                    logger.warning("No se pudo extender frente %s a centro %s: %s", front.id, best_center.id, e)

    logger.info(
        "Asociacion: %d frentes conectados a centros L de %d totales (extendidos: %d)",
        associated, len(collection), extended,
    )
    return collection


def filter_fronts_near_lows(
    collection: FrontCollection,
    centers: list[PressureCenter],
    cfg: AppConfig,
) -> FrontCollection:
    """Filtra frentes que no estén cerca de ningún centro de baja presión.

    Mantiene:
      - frentes ya asociados (center_id)
      - lineas de inestabilidad

    Para el resto: elimina si la distancia mínima a cualquier L es > umbral.
    """
    if not cfg.center_fronts.require_low_center:
        return collection

    lows = [c for c in centers if c.type == "L"]
    if not lows:
        logger.warning("No hay centros L: se mantienen todos los frentes")
        return collection

    max_dist_deg = float(cfg.center_fronts.max_distance_to_low_deg)
    lat_ref = float(getattr(cfg.plotting.projection_params, "central_latitude", 45.0)) if hasattr(cfg, "plotting") else 45.0
    max_dist_km = _deg_to_km_approx(max_dist_deg, lat_ref=lat_ref)

    to_remove: list[str] = []

    for front in collection:
        if front.center_id:
            continue
        if front.front_type == FrontType.INSTABILITY_LINE:
            continue

        # distancia mínima real (km) a cualquier baja
        min_km = float("inf")
        for center in lows:
            d = _front_min_distance_km(front, center)
            if d < min_km:
                min_km = d

        if min_km > max_dist_km:
            to_remove.append(front.id)

    for fid in to_remove:
        collection.remove(fid)

    if to_remove:
        logger.info(
            "Filtro por proximidad a L: eliminados %d frentes lejanos, quedan %d",
            len(to_remove), len(collection),
        )

    return collection


# -----------------------------------------------------------------------------
# Extensión geométrica (igual que tu versión, con pequeños ajustes robustos)
# -----------------------------------------------------------------------------

def smooth_extend_to_center(
    front: Front,
    center: PressureCenter,
    which_end: str,
    cfg: AppConfig,
):
    """Extiende un frente hasta un centro con curva suave tangente.

    Modifica el frente in-place (concatena puntos).
    """
    from mapa_frentes.fronts.connector import _smooth_spline

    c_lat, c_lon = center.lat, _wrap_lon(center.lon)

    if which_end == "start":
        p0_lat, p0_lon = float(front.lats[0]), _wrap_lon(float(front.lons[0]))
        if front.npoints >= 3:
            p1_lat, p1_lon = float(front.lats[2]), _wrap_lon(float(front.lons[2]))
        else:
            p1_lat, p1_lon = float(front.lats[1]), _wrap_lon(float(front.lons[1]))
    else:
        p0_lat, p0_lon = float(front.lats[-1]), _wrap_lon(float(front.lons[-1]))
        if front.npoints >= 3:
            p1_lat, p1_lon = float(front.lats[-3]), _wrap_lon(float(front.lons[-3]))
        else:
            p1_lat, p1_lon = float(front.lats[-2]), _wrap_lon(float(front.lons[-2]))

    dist_km = float(_haversine_km(p0_lat, p0_lon, c_lat, c_lon))
    if dist_km < 10.0:  # si ya está prácticamente encima
        front.center_id = center.id
        front.association_end = which_end
        return

    # Tangente en grados (solo para dirección, no magnitud)
    tan_lat = p0_lat - p1_lat
    tan_lon = _wrap_lon(p0_lon - p1_lon)
    tan_norm = np.sqrt(tan_lat**2 + tan_lon**2)
    if tan_norm > 1e-10:
        tan_lat /= tan_norm
        tan_lon /= tan_norm

    # Distancia en "grados equivalentes" para el control (suave)
    # convertimos km a ~deg a lat media del segmento
    lat_ref = 0.5 * (p0_lat + c_lat)
    km_per_deg = 111.0
    dist_deg = dist_km / km_per_deg

    # Puntos de control Bezier
    cp1_lat = p0_lat + tan_lat * dist_deg / 3.0
    cp1_lon = _wrap_lon(p0_lon + tan_lon * dist_deg / 3.0)

    cp2_lat = c_lat + (p0_lat - c_lat) / 3.0
    cp2_lon = _wrap_lon(c_lon + (_wrap_lon(p0_lon - c_lon)) / 3.0)

    # Evaluar Bezier
    n_pts = max(int(dist_deg / 0.25), 10)
    t = np.linspace(0.0, 1.0, n_pts)
    b0 = (1 - t) ** 3
    b1 = 3 * t * (1 - t) ** 2
    b2 = 3 * t ** 2 * (1 - t)
    b3 = t ** 3

    ext_lats = b0 * p0_lat + b1 * cp1_lat + b2 * cp2_lat + b3 * c_lat
    ext_lons = b0 * p0_lon + b1 * cp1_lon + b2 * cp2_lon + b3 * c_lon
    ext_lons = np.array([_wrap_lon(x) for x in ext_lons], dtype=float)

    # Suavizado spline
    if len(ext_lats) >= 4:
        ext_lats, ext_lons = _smooth_spline(ext_lats, ext_lons, smoothing=0.3)

    # Excluir primer punto (ya existe)
    ext_lats = ext_lats[1:]
    ext_lons = ext_lons[1:]

    if which_end == "start":
        front.lats = np.concatenate([ext_lats[::-1], front.lats])
        front.lons = np.concatenate([ext_lons[::-1], front.lons])
    else:
        front.lats = np.concatenate([front.lats, ext_lats])
        front.lons = np.concatenate([front.lons, ext_lons])

    front.center_id = center.id
    front.association_end = which_end
