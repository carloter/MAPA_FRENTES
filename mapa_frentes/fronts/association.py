"""Asociacion de frentes TFP a centros de baja presion.

Asigna cada frente detectado por TFP al centro L mas cercano
y extiende el extremo mas proximo del frente hasta el centro,
de modo que los frentes "emanan" visualmente de las borrascas.
"""

from __future__ import annotations

import logging

import numpy as np

from mapa_frentes.analysis.pressure_centers import PressureCenter
from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType

logger = logging.getLogger(__name__)


def associate_fronts_to_centers(
    collection: FrontCollection,
    centers: list[PressureCenter],
    cfg: AppConfig,
) -> FrontCollection:
    """Asocia frentes existentes al centro L mas cercano y los extiende.

    Para cada frente:
    1. Calcula la distancia desde cada extremo (inicio/fin) a cada centro L
    2. Si la distancia minima < umbral, asigna center_id
    3. Extiende el extremo mas cercano del frente hasta el centro L,
       insertando puntos intermedios para una curva suave
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

        best_dist = max_dist + 1
        best_center = None
        best_end = None  # "start" o "end"

        for center in lows:
            # Distancia desde el inicio del frente al centro
            d_start = np.sqrt(
                (front.lats[0] - center.lat)**2
                + (front.lons[0] - center.lon)**2
            )
            # Distancia desde el final del frente al centro
            d_end = np.sqrt(
                (front.lats[-1] - center.lat)**2
                + (front.lons[-1] - center.lon)**2
            )

            if d_start < d_end and d_start < best_dist:
                best_dist = d_start
                best_center = center
                best_end = "start"
            elif d_end <= d_start and d_end < best_dist:
                best_dist = d_end
                best_center = center
                best_end = "end"

        if best_dist <= max_dist and best_center is not None:
            front.center_id = best_center.id
            _extend_front_to_center(front, best_center, best_end)
            associated += 1

    logger.info(
        "Asociacion: %d frentes conectados a centros L de %d totales",
        associated, len(collection),
    )
    return collection


def _extend_front_to_center(
    front: Front,
    center: PressureCenter,
    which_end: str,
):
    """Extiende un extremo del frente hasta el centro de presion.

    Inserta puntos intermedios entre el extremo del frente y el centro
    para que la extension sea una curva suave (no un segmento recto brusco).
    """
    c_lat, c_lon = center.lat, center.lon

    if which_end == "start":
        end_lat, end_lon = front.lats[0], front.lons[0]
    else:
        end_lat, end_lon = front.lats[-1], front.lons[-1]

    dist = np.sqrt((end_lat - c_lat)**2 + (end_lon - c_lon)**2)
    if dist < 0.1:
        # Ya esta suficientemente cerca
        return

    # Generar puntos intermedios (1 cada ~0.5 grados)
    n_interp = max(int(dist / 0.5), 2)
    t = np.linspace(0, 1, n_interp + 1)
    # Excluir el ultimo punto (el extremo del frente ya existe)
    t = t[:-1]

    interp_lats = c_lat + t * (end_lat - c_lat)
    interp_lons = c_lon + t * (end_lon - c_lon)

    if which_end == "start":
        # Prepend: centro -> ... -> inicio original -> resto del frente
        front.lats = np.concatenate([interp_lats, front.lats])
        front.lons = np.concatenate([interp_lons, front.lons])
    else:
        # Append: frente -> fin original -> ... -> centro
        # Invertir para que vaya del frente al centro
        front.lats = np.concatenate([front.lats, interp_lats[::-1]])
        front.lons = np.concatenate([front.lons, interp_lons[::-1]])
