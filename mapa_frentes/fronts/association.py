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


def find_nearest_center_for_front(
    front: Front,
    lows: list[PressureCenter],
) -> tuple[PressureCenter | None, str | None, float]:
    """Encuentra el centro L mas cercano a cualquier extremo del frente.

    Returns:
        (center, which_end, distance) donde which_end es "start" o "end".
        Si no hay centros, devuelve (None, None, inf).
    """
    best_dist = float("inf")
    best_center = None
    best_end = None

    for center in lows:
        d_start = np.sqrt(
            (front.lats[0] - center.lat) ** 2
            + (front.lons[0] - center.lon) ** 2
        )
        d_end = np.sqrt(
            (front.lats[-1] - center.lat) ** 2
            + (front.lons[-1] - center.lon) ** 2
        )

        if d_start < d_end and d_start < best_dist:
            best_dist = d_start
            best_center = center
            best_end = "start"
        elif d_end <= d_start and d_end < best_dist:
            best_dist = d_end
            best_center = center
            best_end = "end"

    return best_center, best_end, best_dist


def associate_fronts_to_centers(
    collection: FrontCollection,
    centers: list[PressureCenter],
    cfg: AppConfig,
) -> FrontCollection:
    """Asocia frentes existentes al centro L mas cercano (solo metadata).

    Para cada frente:
    1. Calcula la distancia desde cada extremo (inicio/fin) a cada centro L
    2. Si la distancia minima < umbral, asigna center_id y association_end
    No modifica la geometria del frente.
    """
    max_dist = cfg.center_fronts.max_association_distance_deg
    lows = [c for c in centers if c.type == "L"]

    if not lows:
        return collection

    associated = 0
    for front in collection:
        if front.front_type == FrontType.INSTABILITY_LINE:
            continue
        if front.center_id:
            continue

        best_center, best_end, best_dist = find_nearest_center_for_front(
            front, lows,
        )

        if best_dist <= max_dist and best_center is not None:
            front.center_id = best_center.id
            front.association_end = best_end
            associated += 1

    logger.info(
        "Asociacion: %d frentes conectados a centros L de %d totales",
        associated, len(collection),
    )
    return collection


def smooth_extend_to_center(
    front: Front,
    center: PressureCenter,
    which_end: str,
    cfg: AppConfig,
):
    """Extiende un frente hasta un centro con curva suave tangente.

    Genera una extension tipo Bezier cubica que:
    - Sale tangente al frente en el extremo indicado
    - Curva suavemente hacia el centro L
    - Se suaviza con spline cubica para resultado natural

    Modifica el frente in-place (concatena puntos).
    """
    from mapa_frentes.fronts.connector import _smooth_spline

    c_lat, c_lon = center.lat, center.lon

    if which_end == "start":
        # Tangente: desde punto 1 hacia punto 0 (hacia afuera del frente)
        p0_lat, p0_lon = front.lats[0], front.lons[0]
        if front.npoints >= 3:
            p1_lat, p1_lon = front.lats[2], front.lons[2]
        else:
            p1_lat, p1_lon = front.lats[1], front.lons[1]
    else:
        p0_lat, p0_lon = front.lats[-1], front.lons[-1]
        if front.npoints >= 3:
            p1_lat, p1_lon = front.lats[-3], front.lons[-3]
        else:
            p1_lat, p1_lon = front.lats[-2], front.lons[-2]

    dist = np.sqrt((p0_lat - c_lat) ** 2 + (p0_lon - c_lon) ** 2)
    if dist < 0.1:
        return

    # Tangente del frente en el extremo (apuntando hacia afuera)
    tan_lat = p0_lat - p1_lat
    tan_lon = p0_lon - p1_lon
    tan_norm = np.sqrt(tan_lat ** 2 + tan_lon ** 2)
    if tan_norm > 1e-10:
        tan_lat /= tan_norm
        tan_lon /= tan_norm

    # Puntos de control Bezier cubica:
    # P0 = extremo del frente
    # CP1 = P0 + tangente * dist/3 (continua la direccion del frente)
    # CP2 = centro + (P0 - centro) * 1/3 (curva hacia el centro)
    # P3 = centro
    cp1_lat = p0_lat + tan_lat * dist / 3.0
    cp1_lon = p0_lon + tan_lon * dist / 3.0

    cp2_lat = c_lat + (p0_lat - c_lat) / 3.0
    cp2_lon = c_lon + (p0_lon - c_lon) / 3.0

    # Evaluar Bezier cubica
    n_pts = max(int(dist / 0.3), 8)
    t = np.linspace(0, 1, n_pts)
    b0 = (1 - t) ** 3
    b1 = 3 * t * (1 - t) ** 2
    b2 = 3 * t ** 2 * (1 - t)
    b3 = t ** 3

    ext_lats = b0 * p0_lat + b1 * cp1_lat + b2 * cp2_lat + b3 * c_lat
    ext_lons = b0 * p0_lon + b1 * cp1_lon + b2 * cp2_lon + b3 * c_lon

    # Suavizar la extension con spline
    if len(ext_lats) >= 4:
        ext_lats, ext_lons = _smooth_spline(ext_lats, ext_lons, smoothing=0.3)

    # Concatenar al frente (excluir el primer punto que es el extremo existente)
    ext_lats = ext_lats[1:]
    ext_lons = ext_lons[1:]

    if which_end == "start":
        # Invertir para que vayan del centro al extremo
        front.lats = np.concatenate([ext_lats[::-1], front.lats])
        front.lons = np.concatenate([ext_lons[::-1], front.lons])
    else:
        front.lats = np.concatenate([front.lats, ext_lats])
        front.lons = np.concatenate([front.lons, ext_lons])

    # Actualizar metadata
    front.center_id = center.id
    front.association_end = which_end
