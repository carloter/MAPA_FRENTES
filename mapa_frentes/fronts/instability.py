"""Deteccion de lineas de inestabilidad por convergencia del viento a 850 hPa.

Una linea de inestabilidad es una zona de convergencia del flujo en niveles
bajos que no esta asociada necesariamente a un gradiente termico frontal.
Se manifiesta como una linea de discontinuidad del viento.

Pipeline:
1. Calcular convergencia = -(du/dx + dv/dy) en coordenadas esfericas
2. Suavizar con gaussiano
3. Detectar crestas (maximos locales de convergencia)
4. Reutilizar cluster_and_connect para generar polilineas
"""

import logging

import numpy as np
import xarray as xr

from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.connector import cluster_and_connect
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType
from mapa_frentes.utils.geo import spherical_gradient
from mapa_frentes.utils.smoothing import smooth_field

logger = logging.getLogger(__name__)


def detect_instability_lines(
    ds: xr.Dataset,
    cfg: AppConfig,
) -> list[Front]:
    """Detecta lineas de inestabilidad a partir de convergencia del viento 850 hPa.

    Returns:
        Lista de Front con tipo INSTABILITY_LINE.
    """
    il_cfg = cfg.instability_lines
    if not il_cfg.enabled:
        return []

    from mapa_frentes.fronts.tfp import _remove_time_dim, _ensure_2d

    ds = _remove_time_dim(ds)

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    u850 = _ensure_2d(ds["u850"].values)
    v850 = _ensure_2d(ds["v850"].values)

    # 1. Suavizar viento
    u_smooth = smooth_field(u850, sigma=il_cfg.smooth_sigma)
    v_smooth = smooth_field(v850, sigma=il_cfg.smooth_sigma)

    # 2. Calcular convergencia: -(du/dx + dv/dy)
    du_dx, _ = spherical_gradient(u_smooth, lats, lons)
    _, dv_dy = spherical_gradient(v_smooth, lats, lons)
    convergence = -(du_dx + dv_dy)

    logger.info(
        "Convergencia: min=%.2e, max=%.2e, p95=%.2e",
        np.nanmin(convergence), np.nanmax(convergence),
        np.nanpercentile(convergence, 95),
    )

    # 3. Detectar crestas de convergencia (maximos locales)
    threshold = il_cfg.convergence_threshold
    front_lats, front_lons = _find_convergence_ridges(
        convergence, lats, lons, threshold
    )

    logger.info("Puntos de convergencia encontrados: %d", len(front_lats))

    if len(front_lats) == 0:
        return []

    # 4. Clustering y conexion (reutiliza infraestructura de frentes)
    polylines = cluster_and_connect(
        front_lats, front_lons,
        eps_deg=cfg.tfp.dbscan_eps_deg,
        min_samples=cfg.tfp.dbscan_min_samples,
        min_points=cfg.tfp.min_front_points,
        simplify_tol=cfg.tfp.simplify_tolerance_deg,
        min_front_length_deg=il_cfg.min_length_deg,
        max_hop_deg=cfg.tfp.max_hop_deg,
        angular_weight=cfg.tfp.angular_weight,
        spline_smoothing=cfg.tfp.spline_smoothing,
        merge_distance_deg=cfg.tfp.merge_distance_deg,
    )

    fronts = []
    for i, (plats, plons) in enumerate(polylines):
        front = Front(
            front_type=FrontType.INSTABILITY_LINE,
            lats=plats,
            lons=plons,
            id=f"instab_{i:03d}",
        )
        fronts.append(front)

    logger.info("Lineas de inestabilidad detectadas: %d", len(fronts))
    return fronts


def _find_convergence_ridges(
    convergence: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Encuentra crestas de convergencia (maximos locales por encima del umbral).

    Un punto es cresta si:
    1. convergencia > threshold
    2. Es maximo local en una ventana de 3x3
    """
    ny, nx = convergence.shape
    margin = 3
    ridge_lats = []
    ridge_lons = []

    for j in range(margin, ny - margin):
        for i in range(margin, nx - margin):
            val = convergence[j, i]
            if val <= threshold:
                continue

            # Maximo local en ventana 3x3
            patch = convergence[j - 1:j + 2, i - 1:i + 2]
            if val >= np.max(patch):
                ridge_lats.append(lats[j])
                ridge_lons.append(lons[i])

    return np.array(ridge_lats), np.array(ridge_lons)
