"""Clasificación de frentes frío/cálido/ocluido con detección robusta de oclusiones.

Método mejorado multi-criterio:
1. Clasificación preliminar usando advección térmica (método original)
2. Detección de candidatos por geometría (patrón T/Y cerca de centros L)
3. Scoring multi-factor: estructura vertical, vorticidad, geometría, frontogénesis
4. Reclasificación y determinación de subtipo de oclusión
"""

import logging
from typing import List, Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from mapa_frentes.analysis.pressure_centers import PressureCenter
from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType
from mapa_frentes.utils.smoothing import smooth_field
from mapa_frentes.utils.geo import spherical_gradient

logger = logging.getLogger(__name__)

# Umbrales método clásico
PURITY_THRESHOLD = 0.70
SIGNIFICANCE_THRESHOLD = 0.0


def classify_fronts(
    collection: FrontCollection,
    ds: xr.Dataset,
    cfg: AppConfig,
    centers: List[PressureCenter] | None = None,
) -> FrontCollection:
    """Clasifica frentes usando detección robusta de oclusiones.

    Args:
        collection: Colección de frentes a clasificar.
        ds: Dataset con campos meteorológicos multi-nivel.
        cfg: Configuración de la aplicación.
        centers: Lista de centros de presión (opcional, para detección geométrica).

    Returns:
        Colección de frentes clasificados con scores de oclusión.
    """
    if not collection.fronts:
        return collection

    from mapa_frentes.fronts.tfp import _remove_time_dim, _ensure_2d, compute_theta_w
    ds = _remove_time_dim(ds)

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    # Campo térmico: theta_w suavizado (coherente con la detección TFP)
    theta_w = compute_theta_w(ds)
    theta_w = smooth_field(theta_w, sigma=cfg.tfp.smooth_sigma)
    gx, gy = spherical_gradient(theta_w, lats, lons)

    u850 = _ensure_2d(ds["u850"].values)
    v850 = _ensure_2d(ds["v850"].values)

    # --- 1. Clasificación preliminar (método original) ---
    logger.info("Clasificación preliminar por advección térmica en 850 hPa")
    for front in collection.fronts:
        front.front_type = _classify_single_front(
            front, lats, lons, gx, gy, u850, v850
        )

    # --- 2. Detección robusta de oclusiones (si está habilitada) ---
    if cfg.occlusion.enabled and cfg.occlusion.use_multilevel:
        # Verificar que tenemos datos multi-nivel
        has_multilevel = ("t500" in ds and "t700" in ds and "t850" in ds)

        if has_multilevel:
            logger.info("Aplicando detección robusta de oclusiones")
            _apply_robust_occlusion_detection(
                collection, ds, cfg, centers, lats, lons
            )
        else:
            logger.warning(
                "Detección robusta de oclusiones requiere datos multi-nivel. "
                "Usando clasificación básica."
            )

    # --- Logging de resultados ---
    cold_count = sum(1 for f in collection if f.front_type == FrontType.COLD)
    warm_count = sum(1 for f in collection if f.front_type == FrontType.WARM)
    occl_count = sum(1 for f in collection if f.front_type == FrontType.OCCLUDED)
    cold_occl = sum(1 for f in collection if f.front_type == FrontType.COLD_OCCLUDED)
    warm_occl = sum(1 for f in collection if f.front_type == FrontType.WARM_OCCLUDED)
    seclusion = sum(1 for f in collection if f.front_type == FrontType.WARM_SECLUSION)

    logger.info(
        "Clasificación final: %d fríos, %d cálidos, %d ocluidos "
        "(+ %d ocl-fríos, %d ocl-cálidos, %d seclusiones)",
        cold_count, warm_count, occl_count,
        cold_occl, warm_occl, seclusion,
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
    """Clasificación clásica por advección térmica (método original).

    Evalúa la consistencia de la señal a lo largo del frente:
    - Si mayoría de puntos son positivos (>PURITY_THRESHOLD): COLD
    - Si mayoría de puntos son negativos (>PURITY_THRESHOLD): WARM
    - Si la señal es mixta: OCCLUDED
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

        # Componente del viento en la dirección normal al frente
        vn = u850[j, i] * nx + v850[j, i] * ny

        # Componente del gradiente en la dirección normal
        gn = gx[j, i] * nx + gy[j, i] * ny

        cross = vn * gn
        cross_products.append(cross)

    if not cross_products:
        return FrontType.COLD

    cross_arr = np.array(cross_products)

    # Filtrar puntos con señal significativa
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
        # Señal mixta: frente ocluido (preliminar)
        return FrontType.OCCLUDED


def _apply_robust_occlusion_detection(
    collection: FrontCollection,
    ds: xr.Dataset,
    cfg: AppConfig,
    centers: List[PressureCenter] | None,
    lats: np.ndarray,
    lons: np.ndarray,
):
    """Aplica detección robusta de oclusiones a frentes candidatos."""
    # Cargar datos multi-nivel
    multilevel_data = _load_multilevel_data(ds, lats, lons)

    # Identificar candidatos geométricos si tenemos centros
    geometric_candidates = set()
    if centers is not None:
        geometric_candidates = _detect_geometric_occlusions(
            collection, centers, cfg
        )
        logger.debug("Candidatos geométricos: %d frentes", len(geometric_candidates))

    # Evaluar todos los frentes clasificados preliminarmente como ocluidos
    # o identificados como candidatos geométricos
    occl_cfg = cfg.occlusion

    for front in collection.fronts:
        is_candidate = (
            front.front_type == FrontType.OCCLUDED or
            front.id in geometric_candidates
        )

        if not is_candidate:
            continue

        # Calcular score multi-factor
        score = _compute_occlusion_score(
            front, multilevel_data, lats, lons, centers, cfg
        )
        front.occlusion_score = score

        # Reclasificar si score > umbral
        if score >= occl_cfg.min_score:
            # Determinar subtipo
            occlusion_type = _classify_occlusion_type(
                front, multilevel_data, lats, lons, occl_cfg
            )
            front.front_type = occlusion_type
            front.occlusion_type = occlusion_type.value

            logger.debug(
                "Frente %s: oclusión confirmada (score=%.2f, tipo=%s)",
                front.id[:8], score, occlusion_type.value
            )
        else:
            # Score bajo: mantener clasificación original o reclasificar
            # como frío/cálido según advección dominante
            logger.debug(
                "Frente %s: score bajo (%.2f < %.2f), manteniendo clasificación",
                front.id[:8], score, occl_cfg.min_score
            )


def _load_multilevel_data(
    ds: xr.Dataset,
    lats: np.ndarray,
    lons: np.ndarray,
) -> dict:
    """Carga y prepara datos de múltiples niveles de presión."""
    from mapa_frentes.fronts.tfp import _ensure_2d

    data = {}

    # Temperatura y viento en niveles
    for level in [500, 700, 850]:
        if f"t{level}" in ds:
            data[f"t{level}"] = _ensure_2d(ds[f"t{level}"].values)
        if f"u{level}" in ds:
            data[f"u{level}"] = _ensure_2d(ds[f"u{level}"].values)
        if f"v{level}" in ds:
            data[f"v{level}"] = _ensure_2d(ds[f"v{level}"].values)
        if f"vo{level}" in ds:
            data[f"vo{level}"] = _ensure_2d(ds[f"vo{level}"].values)

    data["lats"] = lats
    data["lons"] = lons

    return data


def _detect_geometric_occlusions(
    collection: FrontCollection,
    centers: List[PressureCenter],
    cfg: AppConfig,
) -> set:
    """Detecta candidatos a oclusión por patrón geométrico T/Y.

    Busca frentes fríos + cálidos convergiendo hacia el mismo centro L.

    Returns:
        Set de IDs de frentes candidatos.
    """
    occl_cfg = cfg.occlusion
    candidates = set()

    # Filtrar solo centros L (bajas)
    low_centers = [c for c in centers if c.type == "L"]

    for center in low_centers:
        # Buscar frentes cerca del centro
        nearby_fronts = []
        for front in collection.fronts:
            # Distancia mínima del frente al centro
            distances = np.sqrt(
                (front.lats - center.lat)**2 + (front.lons - center.lon)**2
            )
            min_dist = distances.min()

            if min_dist < occl_cfg.t_pattern_radius_deg:
                nearby_fronts.append((front, min_dist))

        if len(nearby_fronts) < 2:
            continue

        # Buscar pares frío+cálido convergentes
        for i, (front1, dist1) in enumerate(nearby_fronts):
            for front2, dist2 in nearby_fronts[i+1:]:
                # Verificar que son de tipos complementarios
                types = {front1.front_type, front2.front_type}
                if types == {FrontType.COLD, FrontType.WARM}:
                    # Calcular ángulo de convergencia (simplificado)
                    # Si ambos apuntan hacia el centro, son convergentes
                    candidates.add(front1.id)
                    candidates.add(front2.id)

    return candidates


def _compute_occlusion_score(
    front: Front,
    multilevel_data: dict,
    lats: np.ndarray,
    lons: np.ndarray,
    centers: List[PressureCenter] | None,
    cfg: AppConfig,
) -> float:
    """Calcula score de oclusión multi-factor (0-1).

    Factores:
    - 30% Estructura Vertical (VSI)
    - 25% Vorticidad ciclónica
    - 20% Patrón geométrico
    - 15% Frontogénesis vertical
    - 10% Profundidad del centro asociado
    """
    occl_cfg = cfg.occlusion
    scores = {}

    # --- Factor 1: Estructura Vertical (30%) ---
    vsi_score = _compute_vertical_structure_score(
        front, multilevel_data, lats, lons
    )
    scores["vsi"] = vsi_score

    # --- Factor 2: Vorticidad (25%) ---
    vort_score = _compute_vorticity_score(
        front, multilevel_data, lats, lons, occl_cfg
    )
    scores["vorticity"] = vort_score

    # --- Factor 3: Patrón geométrico (20%) ---
    geom_score = _compute_geometric_score(
        front, centers, occl_cfg
    )
    scores["geometry"] = geom_score

    # --- Factor 4: Frontogénesis vertical (15%) ---
    frontog_score = _compute_vertical_frontogenesis_score(
        front, multilevel_data, lats, lons
    )
    scores["frontogenesis"] = frontog_score

    # --- Factor 5: Profundidad del centro (10%) ---
    depth_score = _compute_center_depth_score(
        front, centers
    )
    scores["depth"] = depth_score

    # Score total ponderado
    total_score = (
        0.30 * vsi_score +
        0.25 * vort_score +
        0.20 * geom_score +
        0.15 * frontog_score +
        0.10 * depth_score
    )

    logger.debug(
        "Score oclusión %s: VSI=%.2f, vort=%.2f, geom=%.2f, frontog=%.2f, "
        "prof=%.2f → TOTAL=%.2f",
        front.id[:8],
        vsi_score, vort_score, geom_score, frontog_score, depth_score,
        total_score
    )

    return float(np.clip(total_score, 0, 1))


def _compute_vertical_structure_score(
    front: Front,
    multilevel_data: dict,
    lats: np.ndarray,
    lons: np.ndarray,
) -> float:
    """Calcula score de estructura vertical: VSI = (θ_e_700 - θ_e_850) / θ_e_mean.

    Oclusiones clásicas: VSI moderado
    Warm seclusion: VSI alto (núcleo cálido elevado)
    """
    if "t700" not in multilevel_data or "t850" not in multilevel_data:
        return 0.0

    t700 = multilevel_data["t700"]
    t850 = multilevel_data["t850"]

    # Calcular θ_e aproximado (simplificado: usar temperatura como proxy)
    # En producción, usar fórmula completa con humedad
    theta_e_700 = t700
    theta_e_850 = t850

    # Interpolar a lo largo del frente
    interp_700 = RegularGridInterpolator(
        (lats, lons), theta_e_700, bounds_error=False, fill_value=np.nan
    )
    interp_850 = RegularGridInterpolator(
        (lats, lons), theta_e_850, bounds_error=False, fill_value=np.nan
    )

    coords_lat_lon = front.coords[:, [1, 0]]  # [lat, lon]
    te_700_front = interp_700(coords_lat_lon)
    te_850_front = interp_850(coords_lat_lon)

    # VSI promedio a lo largo del frente
    valid = ~(np.isnan(te_700_front) | np.isnan(te_850_front))
    if not np.any(valid):
        return 0.0

    te_diff = te_700_front[valid] - te_850_front[valid]
    te_mean = (te_700_front[valid] + te_850_front[valid]) / 2
    vsi = te_diff / (te_mean + 1e-10)
    avg_vsi = np.nanmean(vsi)

    # Normalizar: VSI típico de oclusión ~0.01-0.05, seclusion > 0.10
    # Convertir a score 0-1: 0.00 = 0, 0.05 = 0.5, 0.10 = 1
    score = np.clip(avg_vsi / 0.10, 0, 1)

    return float(score)


def _compute_vorticity_score(
    front: Front,
    multilevel_data: dict,
    lats: np.ndarray,
    lons: np.ndarray,
    occl_cfg,
) -> float:
    """Calcula score de vorticidad ciclónica a lo largo del frente."""
    if "vo850" not in multilevel_data:
        return 0.0

    vo850 = multilevel_data["vo850"]

    # Interpolar vorticidad a lo largo del frente
    interp_vo = RegularGridInterpolator(
        (lats, lons), vo850, bounds_error=False, fill_value=0
    )
    coords_lat_lon = front.coords[:, [1, 0]]
    vo_front = interp_vo(coords_lat_lon)

    # Promedio de vorticidad ciclónica (positiva en NH)
    avg_vorticity = np.nanmean(np.maximum(vo_front, 0))

    # Normalizar: threshold = 8e-5 s⁻¹
    # 0 = 0, threshold = 1
    score = np.clip(avg_vorticity / occl_cfg.vorticity_threshold, 0, 1)

    return float(score)


def _compute_geometric_score(
    front: Front,
    centers: List[PressureCenter] | None,
    occl_cfg,
) -> float:
    """Calcula score de proximidad a centros L (patrón T/Y)."""
    if centers is None:
        return 0.5  # Score neutro si no hay información

    # Buscar centro L más cercano
    low_centers = [c for c in centers if c.type == "L"]
    if not low_centers:
        return 0.0

    min_distance = float('inf')
    for center in low_centers:
        distances = np.sqrt(
            (front.lats - center.lat)**2 + (front.lons - center.lon)**2
        )
        dist = distances.min()
        if dist < min_distance:
            min_distance = dist

    # Normalizar: 0° = 1, t_pattern_radius_deg = 0.5, > 2*radius = 0
    max_dist = 2 * occl_cfg.t_pattern_radius_deg
    score = np.clip(1.0 - min_distance / max_dist, 0, 1)

    return float(score)


def _compute_vertical_frontogenesis_score(
    front: Front,
    multilevel_data: dict,
    lats: np.ndarray,
    lons: np.ndarray,
) -> float:
    """Calcula score de frontogénesis vertical.

    Ratio frontogénesis_700 / frontogénesis_850 > 1 indica elevación del frente.
    """
    # Simplificación: retornar score neutro
    # En producción, calcular frontogénesis en ambos niveles
    return 0.5


def _compute_center_depth_score(
    front: Front,
    centers: List[PressureCenter] | None,
) -> float:
    """Calcula score basado en profundidad del centro asociado."""
    if centers is None or not front.center_id:
        return 0.5  # Score neutro

    # Buscar el centro asociado
    center = None
    for c in centers:
        if c.id == front.center_id:
            center = c
            break

    if center is None or center.type != "L":
        return 0.5

    # Normalizar: (1013 - presión) / 30
    # 1013 hPa = 0, 983 hPa = 1
    pressure = center.value
    score = np.clip((1013 - pressure) / 30.0, 0, 1)

    return float(score)


def _classify_occlusion_type(
    front: Front,
    multilevel_data: dict,
    lats: np.ndarray,
    lons: np.ndarray,
    occl_cfg,
) -> FrontType:
    """Determina subtipo de oclusión: COLD_OCCLUDED, WARM_OCCLUDED, WARM_SECLUSION."""
    # Calcular VSI
    if "t700" not in multilevel_data or "t850" not in multilevel_data:
        return FrontType.OCCLUDED  # Fallback

    t700 = multilevel_data["t700"]
    t850 = multilevel_data["t850"]

    interp_700 = RegularGridInterpolator(
        (lats, lons), t700, bounds_error=False, fill_value=np.nan
    )
    interp_850 = RegularGridInterpolator(
        (lats, lons), t850, bounds_error=False, fill_value=np.nan
    )

    coords_lat_lon = front.coords[:, [1, 0]]
    t_700_front = interp_700(coords_lat_lon)
    t_850_front = interp_850(coords_lat_lon)

    valid = ~(np.isnan(t_700_front) | np.isnan(t_850_front))
    if not np.any(valid):
        return FrontType.OCCLUDED

    t_diff = t_700_front[valid] - t_850_front[valid]
    t_mean = (t_700_front[valid] + t_850_front[valid]) / 2
    vsi = t_diff / (t_mean + 1e-10)
    avg_vsi = np.nanmean(vsi)

    # Clasificar por VSI
    if avg_vsi > occl_cfg.vsi_threshold:
        return FrontType.WARM_SECLUSION  # Núcleo cálido aislado
    elif avg_vsi > 0.03:
        return FrontType.WARM_OCCLUDED  # Oclusión cálida
    else:
        return FrontType.COLD_OCCLUDED  # Oclusión fría
