"""Clasificación de frentes frío/cálido/ocluido con detección robusta de oclusiones.

Método mejorado multi-criterio:
1. Cálculo de cross products punto a punto (advección térmica)
2. Clasificación usando posición relativa al centro L asociado (si existe)
3. Segmentación de frentes mixtos (frio+calido) cerca de centros
4. Detección robusta de oclusiones (estructura vertical, vorticidad, geometría)
"""

import logging
from typing import List

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from mapa_frentes.analysis.pressure_centers import PressureCenter
from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType
from mapa_frentes.utils.smoothing import smooth_field

logger = logging.getLogger(__name__)

# Umbrales
PURITY_THRESHOLD = 0.60          # Bajado de 0.70: 60% basta para asignar tipo
SIGNIFICANCE_THRESHOLD = 0.0
MIXED_THRESHOLD = 0.25           # Si ambos signos superan 25%, considerar segmentar
PROXIMITY_OCCLUDED_DEG = 2.5     # Distancia al centro para marcar segmento ocluido


def classify_fronts(
    collection: FrontCollection,
    ds: xr.Dataset,
    cfg: AppConfig,
    centers: List[PressureCenter] | None = None,
) -> FrontCollection:
    """Clasifica frentes usando advección térmica + geometría del centro.

    Pipeline:
    1. Calcular cross products punto a punto para cada frente
    2. Clasificar usando centro L asociado (si existe) o método clásico
    3. Segmentar frentes mixtos cerca de centros (frio+calido+ocluido)
    4. Detección robusta de oclusiones (multi-nivel, si habilitada)
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

    u850 = _ensure_2d(ds["u850"].values)
    v850 = _ensure_2d(ds["v850"].values)

    # Filtrar centros L
    low_centers = [c for c in (centers or []) if c.type == "L"]

    # --- 1. Calcular señal térmica y clasificar ---
    logger.info("Clasificando frentes por contraste térmico + geometría")
    for front in collection.fronts:
        signals = _compute_thermal_signal(
            front, theta_w, lats, lons, u850, v850
        )
        front.front_type = _classify_with_center(
            front, signals, low_centers
        )

    # --- 2. Segmentar frentes mixtos cerca de centros ---
    if low_centers:
        new_fronts = []
        to_remove = []
        for front in collection.fronts:
            signals = _compute_thermal_signal(
                front, theta_w, lats, lons, u850, v850
            )
            segments = _split_mixed_front(front, signals, low_centers)
            if segments is not None:
                to_remove.append(front.id)
                new_fronts.extend(segments)
                logger.info(
                    "Frente %s segmentado en %d partes: %s",
                    front.id[:8], len(segments),
                    [s.front_type.value for s in segments],
                )

        # Reemplazar frentes originales por segmentos
        for fid in to_remove:
            collection.remove(fid)
        for nf in new_fronts:
            collection.add(nf)

    # --- 3. Detección robusta de oclusiones (si está habilitada) ---
    if cfg.occlusion.enabled and cfg.occlusion.use_multilevel:
        has_multilevel = ("t500" in ds and "t700" in ds and "t850" in ds)
        if has_multilevel:
            logger.info("Aplicando detección robusta de oclusiones")
            _apply_robust_occlusion_detection(
                collection, ds, cfg, centers, lats, lons
            )

    # --- Logging de resultados ---
    counts = {}
    for f in collection:
        t = f.front_type.value
        counts[t] = counts.get(t, 0) + 1
    logger.info("Clasificación final: %s", counts)

    return collection


# ============================================================================
# Cross products: advección térmica punto a punto
# ============================================================================

def _compute_thermal_signal(
    front: Front,
    theta_w: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    u850: np.ndarray,
    v850: np.ndarray,
) -> np.ndarray:
    """Calcula señal térmica con signo en cada punto del frente.

    Método: muestrear theta_w a varias distancias a ambos lados del frente
    (perpendicular) y usar el máximo contraste. El signo lo da la dirección
    del viento: si el viento sopla desde el lado frío → frente frío.

    signal > 0 = frente frío (viento empuja aire frío hacia el lado cálido)
    signal < 0 = frente cálido (viento empuja aire cálido hacia el lado frío)

    Returns:
        Array de señales térmicas, longitud = npoints del frente.
    """
    interp_tw = RegularGridInterpolator(
        (lats, lons), theta_w, bounds_error=False, fill_value=np.nan
    )

    # Muestrear a múltiples distancias para captar gradientes anchos
    sample_distances = [1.0, 2.0, 3.5, 5.0]

    signals = np.zeros(front.npoints)

    for k in range(front.npoints):
        lat, lon = front.lats[k], front.lons[k]

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

        # Normal al frente (perpendicular, apunta hacia la "derecha" del frente)
        nx = -dy / tang_norm
        ny = dx / tang_norm

        cos_lat = max(np.cos(np.radians(lat)), 0.1)

        # Buscar el contraste máximo entre las distancias de muestreo
        best_delta = 0.0
        for dist in sample_distances:
            pt_right = np.array([[lat + ny * dist, lon + nx * dist / cos_lat]])
            pt_left = np.array([[lat - ny * dist, lon - nx * dist / cos_lat]])

            tw_right = interp_tw(pt_right).item()
            tw_left = interp_tw(pt_left).item()

            if np.isnan(tw_right) or np.isnan(tw_left):
                continue

            delta = tw_right - tw_left
            # Quedarse con el contraste de mayor magnitud
            if abs(delta) > abs(best_delta):
                best_delta = delta

        if abs(best_delta) < 0.1:
            # Contraste despreciable (< 0.1 K): sin señal
            continue

        # Componente del viento en la dirección normal
        j = np.argmin(np.abs(lats - lat))
        i = np.argmin(np.abs(lons - lon))
        j = np.clip(j, 0, u850.shape[0] - 1)
        i = np.clip(i, 0, u850.shape[1] - 1)
        vn = u850[j, i] * nx + v850[j, i] * ny

        # Señal normalizada: solo dirección, sin magnitud
        # +1: viento empuja aire frío → frente frío
        # -1: viento empuja aire cálido → frente cálido
        #  0: viento o contraste térmico despreciable
        if abs(vn) > 0.3:
            signals[k] = np.sign(vn * best_delta)
        else:
            # Viento muy débil: usar solo el signo del contraste térmico
            signals[k] = np.sign(best_delta)

    return signals


# ============================================================================
# Clasificación con geometría del centro
# ============================================================================

def _classify_with_center(
    front: Front,
    cross_products: np.ndarray,
    low_centers: List[PressureCenter],
) -> FrontType:
    """Clasifica un frente usando advección térmica + posición relativa al centro L.

    Prioridad:
    1. Si la advección es clara (>PURITY_THRESHOLD) → COLD/WARM
    2. Señal mixta: usar geometría del centro como desempate
    (La proximidad al centro se usa en la segmentación, no aquí)
    """
    significant = cross_products[np.abs(cross_products) > SIGNIFICANCE_THRESHOLD]
    if len(significant) == 0:
        return FrontType.COLD

    n_positive = np.sum(significant > 0)
    n_negative = np.sum(significant < 0)
    n_total = len(significant)
    frac_positive = n_positive / n_total
    frac_negative = n_negative / n_total

    # Señal clara: usarla directamente
    if frac_positive >= PURITY_THRESHOLD:
        return FrontType.COLD
    if frac_negative >= PURITY_THRESHOLD:
        return FrontType.WARM

    # Señal mixta: intentar resolver con geometría del centro
    if low_centers:
        center_type = _classify_by_center_azimuth(front, low_centers)
        if center_type is not None:
            return center_type

    # Sin centro o sin señal clara: usar mayoría simple
    if frac_positive > frac_negative:
        return FrontType.COLD
    elif frac_negative > frac_positive:
        return FrontType.WARM
    else:
        return FrontType.OCCLUDED


def _min_distance_to_nearest_low(
    front: Front,
    low_centers: List[PressureCenter],
) -> float:
    """Distancia mínima de cualquier punto del frente al centro L más cercano."""
    min_dist = float("inf")
    for center in low_centers:
        distances = np.sqrt(
            (front.lats - center.lat) ** 2 + (front.lons - center.lon) ** 2
        )
        d = distances.min()
        if d < min_dist:
            min_dist = d
    return min_dist


def _classify_by_center_azimuth(
    front: Front,
    low_centers: List[PressureCenter],
) -> FrontType | None:
    """Clasifica por posición relativa al centro L más cercano.

    Modelo conceptual de ciclón extratropical (hemisferio norte):
    - Sector S/SW del centro → frente frío (advección fría)
    - Sector E/SE del centro → frente cálido (advección cálida)
    - Muy cerca del centro → ocluido
    """
    # Buscar centro L más cercano
    nearest_center = None
    min_dist = float("inf")
    for center in low_centers:
        distances = np.sqrt(
            (front.lats - center.lat) ** 2 + (front.lons - center.lon) ** 2
        )
        d = distances.min()
        if d < min_dist:
            min_dist = d
            nearest_center = center

    if nearest_center is None or min_dist > 10.0:
        return None  # Demasiado lejos de cualquier centro

    # Si está muy cerca del centro, es candidato a ocluido
    if min_dist < PROXIMITY_OCCLUDED_DEG:
        return FrontType.OCCLUDED

    # Azimut medio del frente respecto al centro
    mid_lat = np.mean(front.lats)
    mid_lon = np.mean(front.lons)
    dlat = mid_lat - nearest_center.lat
    dlon = mid_lon - nearest_center.lon

    # Azimut desde el centro al punto medio del frente (grados desde N, horario)
    azimuth = np.degrees(np.arctan2(dlon, dlat)) % 360.0

    # Modelo conceptual: cold front en sector SW (180-270°), warm front en sector E/SE (45-180°)
    # Pero esto varía mucho, así que solo usamos como desempate suave
    # Sector frío: 150-300° (S/SW/W/NW)
    # Sector cálido: 0-150° o 300-360° (N/NE/E/SE)
    if 150 <= azimuth <= 300:
        return FrontType.COLD
    else:
        return FrontType.WARM


# ============================================================================
# Segmentación de frentes mixtos
# ============================================================================

def _split_mixed_front(
    front: Front,
    cross_products: np.ndarray,
    low_centers: List[PressureCenter],
) -> list[Front] | None:
    """Segmenta un frente cerca de un centro L.

    Dos estrategias:
    A) Por cross products: si hay señal mixta (>25% positivo Y >25% negativo),
       buscar el punto de transición (cambio de signo).
    B) Por azimut: si el frente cruza sectores distintos respecto al centro
       (sector frío vs cálido), partir en el punto más cercano al centro.

    Returns:
        Lista de frentes segmentados, o None si no procede segmentar.
    """
    if front.npoints < 6:
        return None

    # Buscar centro L cercano
    nearest_center = None
    min_dist = float("inf")
    for center in low_centers:
        distances = np.sqrt(
            (front.lats - center.lat) ** 2 + (front.lons - center.lon) ** 2
        )
        d = distances.min()
        if d < min_dist:
            min_dist = d
            nearest_center = center

    if nearest_center is None or min_dist > 10.0:
        return None

    # Calcular azimut de cada punto respecto al centro
    dlats = front.lats - nearest_center.lat
    dlons = front.lons - nearest_center.lon
    azimuths = np.degrees(np.arctan2(dlons, dlats)) % 360.0

    # Sector frío: 150-300° (S/SW/W/NW), sector cálido: resto
    is_cold_sector = (azimuths >= 150) & (azimuths <= 300)
    has_cold_sector = np.any(is_cold_sector)
    has_warm_sector = np.any(~is_cold_sector)

    # --- Estrategia A: cross products mixtos ---
    best_split = None
    significant = cross_products[np.abs(cross_products) > SIGNIFICANCE_THRESHOLD]
    if len(significant) > 0:
        n_total = len(significant)
        frac_positive = np.sum(significant > 0) / n_total
        frac_negative = np.sum(significant < 0) / n_total

        if frac_positive >= MIXED_THRESHOLD and frac_negative >= MIXED_THRESHOLD:
            # Suavizar cross products
            n_smooth = max(3, front.npoints // 8)
            kernel = np.ones(n_smooth) / n_smooth
            cp_smooth = np.convolve(cross_products, kernel, mode="same")

            # Buscar cambios de signo
            sign_changes = []
            for k in range(1, len(cp_smooth)):
                if cp_smooth[k - 1] * cp_smooth[k] < 0:
                    sign_changes.append(k)

            if sign_changes:
                center_distances = np.sqrt(dlats ** 2 + dlons ** 2)
                best_center_dist = float("inf")
                for sc in sign_changes:
                    d = center_distances[sc]
                    if d < best_center_dist:
                        best_center_dist = d
                        best_split = sc

    # --- Estrategia B: cambio de sector azimutal ---
    if best_split is None and has_cold_sector and has_warm_sector:
        # Buscar el punto donde cambia de sector, más cercano al centro
        center_distances = np.sqrt(dlats ** 2 + dlons ** 2)
        sector_changes = []
        for k in range(1, front.npoints):
            if is_cold_sector[k] != is_cold_sector[k - 1]:
                sector_changes.append(k)

        if sector_changes:
            best_center_dist = float("inf")
            for sc in sector_changes:
                d = center_distances[sc]
                if d < best_center_dist:
                    best_center_dist = d
                    best_split = sc

    if best_split is None or best_split < 3 or best_split > front.npoints - 3:
        return None

    # --- Crear segmentos ---
    segments = []
    min_seg_points = 3

    for seg_start, seg_end, seg_idx in [
        (0, best_split + 1, 0),
        (best_split, front.npoints, 1),
    ]:
        seg_lats = front.lats[seg_start:seg_end]
        seg_lons = front.lons[seg_start:seg_end]

        if len(seg_lats) < min_seg_points:
            continue

        seg_cp = cross_products[seg_start:seg_end]
        seg_az = azimuths[seg_start:seg_end]

        # Determinar tipo: usar azimut medio del segmento respecto al centro
        seg_center_dists = np.sqrt(
            (seg_lats - nearest_center.lat) ** 2
            + (seg_lons - nearest_center.lon) ** 2
        )

        if seg_center_dists.min() < PROXIMITY_OCCLUDED_DEG:
            seg_type = FrontType.OCCLUDED
        else:
            # Usar combinación de cross product + azimut
            sig = seg_cp[np.abs(seg_cp) > SIGNIFICANCE_THRESHOLD]
            if len(sig) > 0:
                frac_pos = np.sum(sig > 0) / len(sig)
            else:
                frac_pos = 0.5

            # Azimut medio del segmento
            mean_az = np.mean(seg_az) % 360.0
            in_cold_sector = 150 <= mean_az <= 300

            # Cross product y azimut concuerdan → alta confianza
            # Si no concuerdan → usar el que tenga señal más fuerte
            if frac_pos >= 0.6 or (frac_pos >= 0.4 and in_cold_sector):
                seg_type = FrontType.COLD
            elif frac_pos <= 0.4 or (frac_pos <= 0.6 and not in_cold_sector):
                seg_type = FrontType.WARM
            elif in_cold_sector:
                seg_type = FrontType.COLD
            else:
                seg_type = FrontType.WARM

        seg_front = Front(
            front_type=seg_type,
            lats=seg_lats,
            lons=seg_lons,
            id=f"{front.id}_seg{seg_idx}",
            center_id=front.center_id,
            association_end=front.association_end if seg_idx == 0 else "",
        )
        segments.append(seg_front)

        logger.debug(
            "Segmento %s_%d: %d pts, tipo=%s, az_medio=%.0f°",
            front.id[:8], seg_idx, len(seg_lats), seg_type.value, np.mean(seg_az),
        )

    return segments if len(segments) >= 2 else None


# ============================================================================
# Detección robusta de oclusiones (multi-nivel)
# ============================================================================

def _apply_robust_occlusion_detection(
    collection: FrontCollection,
    ds: xr.Dataset,
    cfg: AppConfig,
    centers: List[PressureCenter] | None,
    lats: np.ndarray,
    lons: np.ndarray,
):
    """Aplica detección robusta de oclusiones a frentes candidatos."""
    multilevel_data = _load_multilevel_data(ds, lats, lons)

    # Identificar candidatos geométricos si tenemos centros
    geometric_candidates = set()
    if centers is not None:
        geometric_candidates = _detect_geometric_occlusions(
            collection, centers, cfg
        )
        logger.debug("Candidatos geométricos: %d frentes", len(geometric_candidates))

    occl_cfg = cfg.occlusion

    for front in collection.fronts:
        is_candidate = (
            front.front_type == FrontType.OCCLUDED
            or front.id in geometric_candidates
        )

        if not is_candidate:
            continue

        score = _compute_occlusion_score(
            front, multilevel_data, lats, lons, centers, cfg
        )
        front.occlusion_score = score

        if score >= occl_cfg.min_score:
            occlusion_type = _classify_occlusion_type(
                front, multilevel_data, lats, lons, occl_cfg
            )
            front.front_type = occlusion_type
            front.occlusion_type = occlusion_type.value
            logger.debug(
                "Frente %s: oclusión confirmada (score=%.2f, tipo=%s)",
                front.id[:8], score, occlusion_type.value,
            )


def _load_multilevel_data(
    ds: xr.Dataset,
    lats: np.ndarray,
    lons: np.ndarray,
) -> dict:
    """Carga y prepara datos de múltiples niveles de presión."""
    from mapa_frentes.fronts.tfp import _ensure_2d

    data = {}
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
    """Detecta candidatos a oclusión por patrón geométrico T/Y."""
    occl_cfg = cfg.occlusion
    candidates = set()

    low_centers = [c for c in centers if c.type == "L"]

    for center in low_centers:
        nearby_fronts = []
        for front in collection.fronts:
            distances = np.sqrt(
                (front.lats - center.lat) ** 2 + (front.lons - center.lon) ** 2
            )
            min_dist = distances.min()
            if min_dist < occl_cfg.t_pattern_radius_deg:
                nearby_fronts.append((front, min_dist))

        if len(nearby_fronts) < 2:
            continue

        # Buscar pares frío+cálido convergentes
        for i, (front1, dist1) in enumerate(nearby_fronts):
            for front2, dist2 in nearby_fronts[i + 1 :]:
                types = {front1.front_type, front2.front_type}
                if types == {FrontType.COLD, FrontType.WARM}:
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
    """Calcula score de oclusión multi-factor (0-1)."""
    occl_cfg = cfg.occlusion

    vsi_score = _compute_vertical_structure_score(front, multilevel_data, lats, lons)
    vort_score = _compute_vorticity_score(front, multilevel_data, lats, lons, occl_cfg)
    geom_score = _compute_geometric_score(front, centers, occl_cfg)
    depth_score = _compute_center_depth_score(front, centers)

    total_score = (
        0.35 * vsi_score
        + 0.25 * vort_score
        + 0.25 * geom_score
        + 0.15 * depth_score
    )

    logger.debug(
        "Score oclusión %s: VSI=%.2f, vort=%.2f, geom=%.2f, prof=%.2f → TOTAL=%.2f",
        front.id[:8], vsi_score, vort_score, geom_score, depth_score, total_score,
    )

    return float(np.clip(total_score, 0, 1))


def _compute_vertical_structure_score(
    front: Front, multilevel_data: dict, lats: np.ndarray, lons: np.ndarray,
) -> float:
    """VSI = (θ_e_700 - θ_e_850) / θ_e_mean."""
    if "t700" not in multilevel_data or "t850" not in multilevel_data:
        return 0.0

    t700 = multilevel_data["t700"]
    t850 = multilevel_data["t850"]

    interp_700 = RegularGridInterpolator(
        (lats, lons), t700, bounds_error=False, fill_value=np.nan
    )
    interp_850 = RegularGridInterpolator(
        (lats, lons), t850, bounds_error=False, fill_value=np.nan
    )

    coords_lat_lon = front.coords[:, [1, 0]]
    te_700_front = interp_700(coords_lat_lon)
    te_850_front = interp_850(coords_lat_lon)

    valid = ~(np.isnan(te_700_front) | np.isnan(te_850_front))
    if not np.any(valid):
        return 0.0

    te_diff = te_700_front[valid] - te_850_front[valid]
    te_mean = (te_700_front[valid] + te_850_front[valid]) / 2
    vsi = te_diff / (te_mean + 1e-10)
    avg_vsi = np.nanmean(vsi)

    return float(np.clip(avg_vsi / 0.10, 0, 1))


def _compute_vorticity_score(
    front: Front, multilevel_data: dict, lats: np.ndarray, lons: np.ndarray, occl_cfg,
) -> float:
    """Score de vorticidad ciclónica a lo largo del frente."""
    if "vo850" not in multilevel_data:
        return 0.0

    vo850 = multilevel_data["vo850"]
    interp_vo = RegularGridInterpolator(
        (lats, lons), vo850, bounds_error=False, fill_value=0
    )
    coords_lat_lon = front.coords[:, [1, 0]]
    vo_front = interp_vo(coords_lat_lon)
    avg_vorticity = np.nanmean(np.maximum(vo_front, 0))

    return float(np.clip(avg_vorticity / occl_cfg.vorticity_threshold, 0, 1))


def _compute_geometric_score(
    front: Front, centers: List[PressureCenter] | None, occl_cfg,
) -> float:
    """Score de proximidad a centros L."""
    if centers is None:
        return 0.5

    low_centers = [c for c in centers if c.type == "L"]
    if not low_centers:
        return 0.0

    min_distance = float("inf")
    for center in low_centers:
        distances = np.sqrt(
            (front.lats - center.lat) ** 2 + (front.lons - center.lon) ** 2
        )
        dist = distances.min()
        if dist < min_distance:
            min_distance = dist

    max_dist = 2 * occl_cfg.t_pattern_radius_deg
    return float(np.clip(1.0 - min_distance / max_dist, 0, 1))


def _compute_center_depth_score(
    front: Front, centers: List[PressureCenter] | None,
) -> float:
    """Score basado en profundidad del centro asociado."""
    if centers is None or not front.center_id:
        return 0.5

    center = None
    for c in centers:
        if c.id == front.center_id:
            center = c
            break

    if center is None or center.type != "L":
        return 0.5

    return float(np.clip((1013 - center.value) / 30.0, 0, 1))


def _classify_occlusion_type(
    front: Front, multilevel_data: dict, lats: np.ndarray, lons: np.ndarray, occl_cfg,
) -> FrontType:
    """Determina subtipo: COLD_OCCLUDED, WARM_OCCLUDED, WARM_SECLUSION."""
    if "t700" not in multilevel_data or "t850" not in multilevel_data:
        return FrontType.OCCLUDED

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

    if avg_vsi > occl_cfg.vsi_threshold:
        return FrontType.WARM_SECLUSION
    elif avg_vsi > 0.03:
        return FrontType.WARM_OCCLUDED
    else:
        return FrontType.COLD_OCCLUDED
