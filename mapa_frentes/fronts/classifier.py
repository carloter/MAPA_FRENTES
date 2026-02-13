"""Clasificacion de frentes frio/calido/ocluido usando viento a 850 hPa.

Criterios:
1. Componente del viento perpendicular al frente (adveccion termica)
2. Consistencia de la senal a lo largo del frente

Un frente es FRIO si el aire frio avanza hacia el lado calido (cross > 0).
Un frente es CALIDO si el aire calido avanza hacia el lado frio (cross < 0).
Un frente es OCLUIDO si la senal es mixta (parte fria y parte calida).
"""

import logging

import numpy as np
import xarray as xr

from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType
from mapa_frentes.utils.smoothing import smooth_field
from mapa_frentes.utils.geo import spherical_gradient

logger = logging.getLogger(__name__)

# Fraccion minima del tipo dominante para clasificar como frio/calido puro.
# Si la fraccion dominante es menor, se clasifica como ocluido.
PURITY_THRESHOLD = 0.70

# Umbral minimo de |cross_product| para considerar un punto significativo.
# Puntos con |cross| < este valor se ignoran (senal debil/ambigua).
SIGNIFICANCE_THRESHOLD = 0.0


def classify_fronts(
    collection: FrontCollection,
    ds: xr.Dataset,
    cfg: AppConfig,
) -> FrontCollection:
    """Clasifica los frentes como frio, calido u ocluido usando viento 850 hPa."""
    if not collection.fronts:
        return collection

    from mapa_frentes.fronts.tfp import _remove_time_dim, _ensure_2d, compute_theta_w
    ds = _remove_time_dim(ds)

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    # Campo termico: theta_w suavizado (coherente con la deteccion TFP)
    theta_w = compute_theta_w(ds)
    theta_w = smooth_field(theta_w, sigma=cfg.tfp.smooth_sigma)
    gx, gy = spherical_gradient(theta_w, lats, lons)

    u850 = _ensure_2d(ds["u850"].values)
    v850 = _ensure_2d(ds["v850"].values)

    for front in collection.fronts:
        front.front_type = _classify_single_front(
            front, lats, lons, gx, gy, u850, v850
        )

    cold_count = sum(1 for f in collection if f.front_type == FrontType.COLD)
    warm_count = sum(1 for f in collection if f.front_type == FrontType.WARM)
    occl_count = sum(1 for f in collection if f.front_type == FrontType.OCCLUDED)
    logger.info(
        "Clasificacion: %d frios, %d calidos, %d ocluidos",
        cold_count, warm_count, occl_count,
    )

    return collection


def _classify_single_front(
    front: Front,
    lats: np.ndarray,
    lons: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    u850: np.ndarray,
    v850: np.ndarray,
) -> FrontType:
    """Clasifica un frente evaluando la adveccion termica.

    Evalua la consistencia de la senal a lo largo del frente:
    - Si la mayoria de puntos son positivos (>PURITY_THRESHOLD): COLD
    - Si la mayoria de puntos son negativos (>PURITY_THRESHOLD): WARM
    - Si la senal es mixta: OCCLUDED
    """
    cross_products = []

    for k in range(len(front.lats)):
        lat, lon = front.lats[k], front.lons[k]

        j = np.argmin(np.abs(lats - lat))
        i = np.argmin(np.abs(lons - lon))
        j = np.clip(j, 0, gx.shape[0] - 1)
        i = np.clip(i, 0, gx.shape[1] - 1)

        # Tangente al frente en este punto
        if k == 0:
            dx = front.lons[min(k + 1, front.npoints - 1)] - lon
            dy = front.lats[min(k + 1, front.npoints - 1)] - lat
        elif k == front.npoints - 1:
            dx = lon - front.lons[k - 1]
            dy = lat - front.lats[k - 1]
        else:
            dx = front.lons[k + 1] - front.lons[k - 1]
            dy = front.lats[k + 1] - front.lats[k - 1]

        tang_norm = np.sqrt(dx**2 + dy**2)
        if tang_norm < 1e-10:
            continue

        # Normal al frente (perpendicular a la tangente)
        nx = -dy / tang_norm
        ny = dx / tang_norm

        # Componente del viento en la direccion normal al frente
        vn = u850[j, i] * nx + v850[j, i] * ny

        # Componente del gradiente en la direccion normal
        gn = gx[j, i] * nx + gy[j, i] * ny

        cross = vn * gn
        cross_products.append(cross)

    if not cross_products:
        return FrontType.COLD

    cross_arr = np.array(cross_products)

    # Filtrar puntos con senal significativa
    significant = cross_arr[np.abs(cross_arr) > SIGNIFICANCE_THRESHOLD]
    if len(significant) == 0:
        return FrontType.COLD

    n_positive = np.sum(significant > 0)
    n_negative = np.sum(significant < 0)
    n_total = len(significant)

    frac_positive = n_positive / n_total
    frac_negative = n_negative / n_total

    if frac_positive >= PURITY_THRESHOLD:
        return FrontType.COLD
    elif frac_negative >= PURITY_THRESHOLD:
        return FrontType.WARM
    else:
        # Senal mixta: frente ocluido
        return FrontType.OCCLUDED
