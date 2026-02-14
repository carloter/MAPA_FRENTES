"""Ranking de frentes principales vs secundarios dentro de sistemas ciclónicos."""

import logging
from typing import List

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.models import Front, FrontType, CycloneSystem, CycloneSystemCollection
from mapa_frentes.utils.geo import spherical_gradient
from mapa_frentes.utils.smoothing import smooth_field

logger = logging.getLogger(__name__)


def compute_front_importance(
    front: Front,
    system: CycloneSystem,
    ds: xr.Dataset,
    cfg: AppConfig,
) -> float:
    """Calcula un score de importancia (0-1) para un frente dentro de su sistema.

    Score compuesto ponderado:
    - 25% Longitud del frente
    - 25% Intensidad térmica (gradiente)
    - 20% Frontogénesis activa
    - 15% Distancia al centro del ciclón
    - 10% Tipo de frente (según madurez del ciclón)
    - 5% Profundidad del centro

    Args:
        front: Frente a evaluar.
        system: Sistema ciclónico al que pertenece.
        ds: Dataset con campos meteorológicos.
        cfg: Configuración.

    Returns:
        Score de importancia entre 0 y 1.
    """
    from mapa_frentes.fronts.tfp import _remove_time_dim, _ensure_2d, compute_theta_w

    ds = _remove_time_dim(ds)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    # --- Factor 1: Longitud (25%) ---
    length_deg = _polyline_length_deg(front.coords)
    # Normalizar: 5° = 0, 20° = 1
    length_score = np.clip((length_deg - 5.0) / 15.0, 0, 1)

    # --- Factor 2: Intensidad térmica (25%) ---
    theta_w = compute_theta_w(ds)
    theta_w = smooth_field(theta_w, sigma=cfg.tfp.smooth_sigma)
    gx, gy = spherical_gradient(theta_w, lats, lons)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Interpolar gradiente a lo largo del frente
    interp_grad = RegularGridInterpolator(
        (lats, lons), grad_mag, bounds_error=False, fill_value=0
    )
    front_grads = interp_grad(front.coords[:, [1, 0]])  # [lat, lon]
    avg_grad = np.nanmean(front_grads)

    # Normalizar: 3e-6 = 0, 8e-6 = 1
    thermal_score = np.clip((avg_grad - 3e-6) / 5e-6, 0, 1)

    # --- Factor 3: Frontogénesis (20%) ---
    frontogenesis_score = 0.0
    if "u850" in ds and "v850" in ds:
        u850 = _ensure_2d(ds["u850"].values)
        v850 = _ensure_2d(ds["v850"].values)

        # Calcular frontogénesis simplificada: F ≈ |∇θ| * convergence
        div_u_x, _ = spherical_gradient(u850, lats, lons)
        _, div_v_y = spherical_gradient(v850, lats, lons)
        convergence = -(div_u_x + div_v_y)

        frontogenesis = grad_mag * convergence
        interp_frontog = RegularGridInterpolator(
            (lats, lons), frontogenesis, bounds_error=False, fill_value=0
        )
        front_frontog = interp_frontog(front.coords[:, [1, 0]])
        avg_frontog = np.nanmean(np.maximum(front_frontog, 0))  # Solo valores positivos

        # Normalizar: 0 = 0, 5e-10 = 1
        frontogenesis_score = np.clip(avg_frontog / 5e-10, 0, 1)

    # --- Factor 4: Distancia al centro (15%) ---
    center_lat = system.center.lat
    center_lon = system.center.lon
    distances = np.sqrt(
        (front.lats - center_lat)**2 + (front.lons - center_lon)**2
    )
    min_dist = distances.min()

    # Normalizar: 0° = 1, 10° = 0 (más cerca = más importante)
    distance_score = np.clip(1.0 - min_dist / 10.0, 0, 1)

    # --- Factor 5: Tipo de frente según madurez del ciclón (10%) ---
    pressure = system.center.value
    type_score = 0.5  # Default

    if pressure > 990:
        # Ciclón en desarrollo: COLD > WARM > OCCLUDED
        if front.front_type == FrontType.COLD:
            type_score = 1.0
        elif front.front_type == FrontType.WARM:
            type_score = 0.7
        else:  # Occluded
            type_score = 0.4
    else:
        # Ciclón maduro: OCCLUDED > COLD > WARM
        if front.front_type in (FrontType.OCCLUDED, FrontType.COLD_OCCLUDED,
                                FrontType.WARM_OCCLUDED, FrontType.WARM_SECLUSION):
            type_score = 1.0
        elif front.front_type == FrontType.COLD:
            type_score = 0.7
        else:  # Warm
            type_score = 0.4

    # --- Factor 6: Profundidad del centro (5%) ---
    # Normalizar: (1013 - presión) / 63
    # 1013 hPa = 0, 950 hPa = 1
    depth_score = np.clip((1013 - pressure) / 63.0, 0, 1)

    # --- Score final ponderado ---
    total_score = (
        0.25 * length_score +
        0.25 * thermal_score +
        0.20 * frontogenesis_score +
        0.15 * distance_score +
        0.10 * type_score +
        0.05 * depth_score
    )

    logger.debug(
        "Frente %s: L=%.1f° (%.2f), ∇θ=%.1e (%.2f), F=%.2f, d=%.1f° (%.2f), "
        "tipo=%.2f, prof=%.0fhPa (%.2f) → TOTAL=%.2f",
        front.id[:8],
        length_deg, length_score,
        avg_grad, thermal_score,
        frontogenesis_score,
        min_dist, distance_score,
        type_score,
        pressure, depth_score,
        total_score
    )

    return float(np.clip(total_score, 0, 1))


def rank_fronts_in_system(
    system: CycloneSystem,
    ds: xr.Dataset,
    cfg: AppConfig,
    threshold: float = 0.60,
):
    """Clasifica frentes de un sistema como principales o secundarios.

    Args:
        system: Sistema ciclónico.
        ds: Dataset con campos meteorológicos.
        cfg: Configuración.
        threshold: Score mínimo para clasificar como principal (default: 0.60).
    """
    if not system.fronts:
        return

    for front in system.fronts:
        score = compute_front_importance(front, system, ds, cfg)
        front.importance_score = score
        front.is_primary = (score >= threshold)

    # Ordenar frentes por score descendente
    system.fronts.sort(key=lambda f: f.importance_score, reverse=True)

    n_primary = sum(1 for f in system.fronts if f.is_primary)
    logger.info(
        "Sistema %s: %d frentes primarios, %d secundarios",
        system.id, n_primary, len(system.fronts) - n_primary
    )


def rank_all_systems(
    systems: CycloneSystemCollection,
    ds: xr.Dataset,
    cfg: AppConfig,
    threshold: float = 0.60,
):
    """Aplica ranking a todos los sistemas de la colección.

    Args:
        systems: Colección de sistemas ciclónicos.
        ds: Dataset con campos meteorológicos.
        cfg: Configuración.
        threshold: Score mínimo para clasificar como principal.
    """
    logger.info("Aplicando ranking a %d sistemas", len(systems))

    for system in systems:
        rank_fronts_in_system(system, ds, cfg, threshold)


def _polyline_length_deg(coords: np.ndarray) -> float:
    """Calcula longitud de una polilínea en grados (aproximación euclidiana).

    Args:
        coords: Array (N, 2) con [lon, lat].

    Returns:
        Longitud total en grados.
    """
    if len(coords) < 2:
        return 0.0

    diffs = np.diff(coords, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    return float(seg_lengths.sum())
