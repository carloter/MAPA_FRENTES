"""Agrupado de frentes por sistema ciclónico."""

import logging
from typing import List

import numpy as np

from mapa_frentes.analysis.pressure_centers import PressureCenter
from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.models import (
    Front,
    FrontCollection,
    CycloneSystem,
    CycloneSystemCollection,
)

logger = logging.getLogger(__name__)


def build_cyclone_systems(
    fronts: FrontCollection | List[Front],
    centers: List[PressureCenter],
    cfg: AppConfig,
) -> CycloneSystemCollection:
    """Construye sistemas ciclónicos agrupando frentes por centro L.

    Args:
        fronts: Colección de frentes o lista de frentes.
        centers: Lista de centros de presión detectados.
        cfg: Configuración de la aplicación.

    Returns:
        CycloneSystemCollection con sistemas organizados.
    """
    if isinstance(fronts, FrontCollection):
        front_list = list(fronts.fronts)
        valid_time = fronts.valid_time
    else:
        front_list = fronts
        valid_time = ""

    # 1. Separar centros L (bajas) en primarios y secundarios
    low_centers = [c for c in centers if c.type == "L"]
    primary_lows = [c for c in low_centers if c.primary]
    secondary_lows = [c for c in low_centers if not c.primary]

    logger.info(
        "Construyendo sistemas: %d centros L primarios, %d secundarios",
        len(primary_lows), len(secondary_lows)
    )

    systems = []
    assigned_fronts = set()

    # 2. Para cada centro L primario, crear un sistema
    for primary_center in primary_lows:
        system = CycloneSystem(
            center=primary_center,
            name=primary_center.name,
            valid_time=valid_time,
        )

        # 2a. Añadir frentes directamente asociados al centro primario
        for front in front_list:
            if front.center_id == primary_center.id:
                system.fronts.append(front)
                assigned_fronts.add(front.id)

        # 2b. Buscar centros secundarios cercanos
        secondary_radius = cfg.pressure_centers.secondary_radius_deg
        for sec_center in secondary_lows:
            dist = _haversine_distance(
                primary_center.lat, primary_center.lon,
                sec_center.lat, sec_center.lon
            )
            if dist < secondary_radius:
                system.secondary_centers.append(sec_center)

                # Añadir frentes del secundario al sistema primario
                for front in front_list:
                    if front.center_id == sec_center.id and front.id not in assigned_fronts:
                        system.fronts.append(front)
                        assigned_fronts.add(front.id)

        # 2c. Calcular estadísticas del sistema
        system.compute_statistics()
        systems.append(system)

        logger.info(
            "Sistema %s (%s): %d frentes, %.1f hPa, %d centros secundarios",
            system.id,
            system.name or "sin nombre",
            len(system.fronts),
            system.center.value,
            len(system.secondary_centers)
        )

    # 3. Identificar frentes no asociados
    unassociated = [f for f in front_list if f.id not in assigned_fronts]

    logger.info(
        "Sistemas construidos: %d sistemas, %d frentes no asociados",
        len(systems), len(unassociated)
    )

    return CycloneSystemCollection(
        systems=systems,
        unassociated_fronts=unassociated,
        valid_time=valid_time,
    )


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distancia haversine aproximada en grados."""
    # Aproximación simple: distancia euclidiana en coordenadas esféricas
    # Suficiente para filtrar por radio, no necesita precisión exacta
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return np.sqrt(dlat**2 + dlon**2)
