"""Clasificación de frentes frío/cálido/ocluido (Hewson 1998).

Método:
1. Front speed (Hewson 1998, Eq. 4): V · (∇|∇θ_w| / |∇|∇θ_w||)
   - speed < -K3 → COLD (frente empujado hacia aire cálido)
   - speed > +K3 → WARM (frente empujado hacia aire frío)
   - |speed| ≤ K3 → STATIONARY
2. Segmentación de frentes mixtos cerca de centros L
3. Detección robusta de oclusiones (estructura vertical multi-nivel)
"""

import logging
from typing import List

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from mapa_frentes.analysis.pressure_centers import PressureCenter
from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType

logger = logging.getLogger(__name__)

# Umbrales Hewson/Berry
K3_FRONT_SPEED = 1.5             # m/s: umbral cold/warm vs stationary (Berry et al. 2011)
PROXIMITY_OCCLUDED_DEG = 2.5     # grados: distancia al centro para marcar ocluido


def classify_fronts(
    collection: FrontCollection,
    ds: xr.Dataset,
    cfg: AppConfig,
    centers: List[PressureCenter] | None = None,
) -> FrontCollection:
    """Clasifica frentes usando front speed de Hewson (1998).

    Pipeline:
    1. Calcular front_speed (Eq. 4) en cada punto del frente
    2. Clasificar: cold / warm / stationary
    3. Segmentar frentes mixtos cerca de centros L
    4. Detección robusta de oclusiones (multi-nivel, si habilitada)
    """
    if not collection.fronts:
        return collection

    from mapa_frentes.fronts.tfp import _remove_time_dim, _ensure_2d
    ds = _remove_time_dim(ds)

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    u850 = _ensure_2d(ds["u850"].values)
    v850 = _ensure_2d(ds["v850"].values)

    # Obtener ∇|∇θ_w| del metadata de la collection (calculado en tfp.py)
    gmag_x = collection.metadata.get("gmag_x")
    gmag_y = collection.metadata.get("gmag_y")
    meta_lats = collection.metadata.get("lats")
    meta_lons = collection.metadata.get("lons")

    low_centers = [c for c in (centers or []) if c.type == "L"]

    if gmag_x is not None and gmag_y is not None:
        # --- Método Hewson: front_speed ---
        logger.info("Clasificando frentes por front_speed (Hewson 1998 Eq. 4)")

        # Usar lats/lons del metadata (coherentes con el campo TFP)
        field_lats = meta_lats if meta_lats is not None else lats
        field_lons = meta_lons if meta_lons is not None else lons

        # Interpoladores (se crean una vez, se usan para todos los frentes)
        interp_gmx = RegularGridInterpolator(
            (field_lats, field_lons), gmag_x,
            bounds_error=False, fill_value=0,
        )
        interp_gmy = RegularGridInterpolator(
            (field_lats, field_lons), gmag_y,
            bounds_error=False, fill_value=0,
        )
        interp_u = RegularGridInterpolator(
            (lats, lons), u850, bounds_error=False, fill_value=0,
        )
        interp_v = RegularGridInterpolator(
            (lats, lons), v850, bounds_error=False, fill_value=0,
        )

        # 1. Clasificar cada frente
        for front in collection.fronts:
            if front.front_type == FrontType.INSTABILITY_LINE:
                continue
            speeds = _compute_hewson_front_speed(
                front, interp_gmx, interp_gmy, interp_u, interp_v,
            )
            front.front_type = _classify_from_speed(speeds)

        # 2. Segmentar frentes mixtos cerca de centros L
        if low_centers:
            new_fronts = []
            to_remove = []
            for front in collection.fronts:
                if front.front_type == FrontType.INSTABILITY_LINE:
                    continue
                speeds = _compute_hewson_front_speed(
                    front, interp_gmx, interp_gmy, interp_u, interp_v,
                )
                segments = _split_mixed_front_by_speed(
                    front, speeds, low_centers,
                )
                if segments is not None:
                    to_remove.append(front.id)
                    new_fronts.extend(segments)
                    logger.info(
                        "Frente %s segmentado en %d partes: %s",
                        front.id[:8], len(segments),
                        [s.front_type.value for s in segments],
                    )

            for fid in to_remove:
                collection.remove(fid)
            for nf in new_fronts:
                collection.add(nf)
    else:
        logger.warning(
            "No hay campos gmag_x/gmag_y en metadata. "
            "Clasificacion por azimut del centro (fallback)."
        )
        for front in collection.fronts:
            if front.front_type == FrontType.INSTABILITY_LINE:
                continue
            if low_centers:
                ftype = _classify_by_center_azimuth(front, low_centers)
                if ftype is not None:
                    front.front_type = ftype
            # else: mantener COLD por defecto

    # --- 3. Detección robusta de oclusiones (si habilitada) ---
    if cfg.occlusion.enabled and cfg.occlusion.use_multilevel:
        has_multilevel = ("t500" in ds and "t700" in ds and "t850" in ds)
        if has_multilevel:
            logger.info("Aplicando detección robusta de oclusiones")
            _apply_robust_occlusion_detection(
                collection, ds, cfg, centers, lats, lons
            )

    # --- Logging ---
    counts = {}
    for f in collection:
        t = f.front_type.value
        counts[t] = counts.get(t, 0) + 1
    logger.info("Clasificación final: %s", counts)

    return collection


# ============================================================================
# Front speed (Hewson 1998, Eq. 4)
# ============================================================================

def _compute_hewson_front_speed(
    front: Front,
    interp_gmx: RegularGridInterpolator,
    interp_gmy: RegularGridInterpolator,
    interp_u: RegularGridInterpolator,
    interp_v: RegularGridInterpolator,
) -> np.ndarray:
    """Calcula front_speed (Hewson 1998 Eq. 4) en cada punto del frente.

    front_speed = V · (∇|∇θ_w| / |∇|∇θ_w||)

    Donde ∇|∇θ_w| apunta perpendicular al frente, hacia la zona donde
    el gradiente térmico crece (hacia la zona baroclina).

    Negativo = cold front (viento empuja contorno hacia aire cálido)
    Positivo = warm front (viento empuja contorno hacia aire frío)
    """
    speeds = np.zeros(front.npoints)
    pts = np.column_stack([front.lats, front.lons])

    # Interpolacion vectorizada
    gx_vals = interp_gmx(pts)
    gy_vals = interp_gmy(pts)
    u_vals = interp_u(pts)
    v_vals = interp_v(pts)

    gm = np.sqrt(gx_vals**2 + gy_vals**2)
    valid = gm > 1e-15

    if np.any(valid):
        nx = np.where(valid, gx_vals / np.where(valid, gm, 1.0), 0.0)
        ny = np.where(valid, gy_vals / np.where(valid, gm, 1.0), 0.0)
        speeds = u_vals * nx + v_vals * ny

    return speeds


def _classify_from_speed(
    front_speeds: np.ndarray,
    k3: float = K3_FRONT_SPEED,
) -> FrontType:
    """Clasifica un frente por su velocidad media (Hewson 1998).

    front_speed < -K3 → COLD (contorno se desplaza hacia aire cálido)
    front_speed > +K3 → WARM (contorno se desplaza hacia aire frío)
    |front_speed| ≤ K3 → STATIONARY
    """
    valid = front_speeds != 0
    if not np.any(valid):
        return FrontType.COLD  # fallback

    mean_speed = np.mean(front_speeds[valid])

    if mean_speed < -k3:
        return FrontType.COLD
    elif mean_speed > k3:
        return FrontType.WARM
    else:
        return FrontType.STATIONARY


# ============================================================================
# Segmentación de frentes mixtos por front_speed
# ============================================================================

def _split_mixed_front_by_speed(
    front: Front,
    front_speeds: np.ndarray,
    low_centers: List[PressureCenter],
) -> list[Front] | None:
    """Segmenta un frente cerca de un centro L usando front_speed.

    Si el frente tiene puntos con speed < -K3 (frio) Y puntos con speed > +K3
    (calido), busca el punto de transicion para segmentar.
    """
    if front.npoints < 8:
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

    # Suavizar front_speed para evitar cambios espurios
    n_smooth = max(3, front.npoints // 8)
    kernel = np.ones(n_smooth) / n_smooth
    speed_smooth = np.convolve(front_speeds, kernel, mode="same")

    # Comprobar si hay señal mixta
    has_cold = np.any(speed_smooth < -K3_FRONT_SPEED)
    has_warm = np.any(speed_smooth > K3_FRONT_SPEED)
    if not (has_cold and has_warm):
        return None

    # Buscar cambio de signo en speed_smooth
    sign_changes = []
    for k in range(1, len(speed_smooth)):
        if speed_smooth[k - 1] * speed_smooth[k] < 0:
            sign_changes.append(k)

    if not sign_changes:
        return None

    # Elegir el cambio de signo más cercano al centro
    dlats = front.lats - nearest_center.lat
    dlons = front.lons - nearest_center.lon
    center_distances = np.sqrt(dlats**2 + dlons**2)

    best_split = None
    best_center_dist = float("inf")
    for sc in sign_changes:
        d = center_distances[sc]
        if d < best_center_dist:
            best_center_dist = d
            best_split = sc

    if best_split is None or best_split < 4 or best_split > front.npoints - 4:
        return None

    # Crear segmentos
    segments = []
    for seg_start, seg_end, seg_idx in [
        (0, best_split + 1, 0),
        (best_split, front.npoints, 1),
    ]:
        seg_lats = front.lats[seg_start:seg_end]
        seg_lons = front.lons[seg_start:seg_end]
        if len(seg_lats) < 4:
            continue

        seg_speeds = front_speeds[seg_start:seg_end]

        # Distancia minima del segmento al centro
        seg_dists = np.sqrt(
            (seg_lats - nearest_center.lat)**2
            + (seg_lons - nearest_center.lon)**2
        )

        if seg_dists.min() < PROXIMITY_OCCLUDED_DEG:
            seg_type = FrontType.OCCLUDED
        else:
            seg_type = _classify_from_speed(seg_speeds)

        seg_front = Front(
            front_type=seg_type,
            lats=seg_lats,
            lons=seg_lons,
            id=f"{front.id}_seg{seg_idx}",
            center_id=front.center_id,
            association_end=front.association_end if seg_idx == 0 else "",
        )
        segments.append(seg_front)

    return segments if len(segments) >= 2 else None


# ============================================================================
# Fallback: clasificación por azimut (solo cuando no hay campos Hewson)
# ============================================================================

def _classify_by_center_azimuth(
    front: Front,
    low_centers: List[PressureCenter],
) -> FrontType | None:
    """Clasifica por posición relativa al centro L más cercano.

    Modelo conceptual de ciclón extratropical (hemisferio norte):
    - Sector S/SW del centro → frente frío
    - Sector E/SE del centro → frente cálido
    - Muy cerca del centro → ocluido
    """
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

    if nearest_center is None or min_dist > 8.0:
        return None

    if min_dist < PROXIMITY_OCCLUDED_DEG:
        return FrontType.OCCLUDED

    mid_lat = np.mean(front.lats)
    mid_lon = np.mean(front.lons)
    dlat = mid_lat - nearest_center.lat
    dlon = mid_lon - nearest_center.lon
    azimuth = np.degrees(np.arctan2(dlon, dlat)) % 360.0

    if 180 <= azimuth <= 280:
        return FrontType.COLD
    elif 30 <= azimuth <= 150:
        return FrontType.WARM
    else:
        return None


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


