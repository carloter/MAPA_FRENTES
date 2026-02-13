"""Generacion de frentes meteorologicos desde centros de baja presion.

Algoritmo:
1. Calcular theta_w y su gradiente alrededor del centro L
2. Scoring azimutal: muestrear adveccion termica en un circulo
   alrededor del centro para encontrar las direcciones de frente frio
   (adveccion fria, tipicamente S/SW) y calido (adveccion calida, E/NE)
3. Ray-marching guiado por gradiente: trazar cada frente desde el centro
   hacia afuera siguiendo la cresta del gradiente termico
4. Deteccion de oclusion: si existen ambos frentes, buscar segmento
   ocluido en el sector intermedio (tipicamente NW)
5. Suavizar con spline y crear objetos Front
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from mapa_frentes.analysis.pressure_centers import PressureCenter
from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.models import Front, FrontType
from mapa_frentes.utils.geo import spherical_gradient
from mapa_frentes.utils.smoothing import smooth_field

logger = logging.getLogger(__name__)


def generate_fronts_from_center(
    center: PressureCenter,
    ds,
    cfg: AppConfig,
) -> list[Front]:
    """Genera frentes frio, calido y (opcionalmente) ocluido desde un centro L.

    Args:
        center: Centro de baja presion.
        ds: Dataset xarray con variables t850, q850, u850, v850, msl.
        cfg: Configuracion de la aplicacion.

    Returns:
        Lista de 0-3 objetos Front (frio, calido, ocluido).
    """
    cf_cfg = cfg.center_fronts

    # 1. Preparar campos
    theta_w, gx, gy, grad_mag, u850, v850, lats, lons = _prepare_fields(
        center, ds, cfg,
    )
    if theta_w is None:
        logger.warning("No se pudieron preparar campos para centro %s", center.id)
        return []

    # 2. Scoring azimutal
    azimuths, scores = _azimuthal_advection_scores(
        center.lat, center.lon,
        theta_w, gx, gy, grad_mag, u850, v850,
        lats, lons,
        radius_deg=cf_cfg.search_radius_deg,
    )

    # 3. Encontrar direcciones
    cold_az = _find_front_direction(azimuths, scores, sign=+1)
    warm_az = _find_front_direction(azimuths, scores, sign=-1)

    logger.info(
        "Centro %s (%.1fN %.1f%s): cold_az=%s, warm_az=%s",
        center.id, center.lat, abs(center.lon),
        "W" if center.lon < 0 else "E",
        f"{cold_az:.0f}" if cold_az is not None else "None",
        f"{warm_az:.0f}" if warm_az is not None else "None",
    )

    fronts = []

    # 4. Trazar frente frio
    if cold_az is not None:
        cold_pts = _trace_front_ray(
            center.lat, center.lon, cold_az,
            grad_mag, gx, gy, lats, lons,
            max_length_deg=cf_cfg.max_front_length_deg,
            step_deg=cf_cfg.trace_step_deg,
            max_turn_deg=cf_cfg.max_turn_deg,
            gradient_cutoff_factor=cf_cfg.gradient_cutoff_factor,
        )
        if cold_pts is not None:
            front = _smooth_and_create_front(
                cold_pts[0], cold_pts[1],
                FrontType.COLD, center.id,
                cf_cfg.spline_smoothing,
            )
            fronts.append(front)
            logger.info("Frente frio generado: %d puntos", front.npoints)

    # 5. Trazar frente calido
    if warm_az is not None:
        warm_pts = _trace_front_ray(
            center.lat, center.lon, warm_az,
            grad_mag, gx, gy, lats, lons,
            max_length_deg=cf_cfg.max_front_length_deg,
            step_deg=cf_cfg.trace_step_deg,
            max_turn_deg=cf_cfg.max_turn_deg,
            gradient_cutoff_factor=cf_cfg.gradient_cutoff_factor,
        )
        if warm_pts is not None:
            front = _smooth_and_create_front(
                warm_pts[0], warm_pts[1],
                FrontType.WARM, center.id,
                cf_cfg.spline_smoothing,
            )
            fronts.append(front)
            logger.info("Frente calido generado: %d puntos", front.npoints)

    # 6. Frente ocluido (entre frio y calido, pasando por NW)
    if cold_az is not None and warm_az is not None:
        occl_az = _occlusion_azimuth(cold_az, warm_az)
        if occl_az is not None:
            occl_pts = _trace_front_ray(
                center.lat, center.lon, occl_az,
                grad_mag, gx, gy, lats, lons,
                max_length_deg=cf_cfg.max_front_length_deg * 0.7,
                step_deg=cf_cfg.trace_step_deg,
                max_turn_deg=cf_cfg.max_turn_deg,
                gradient_cutoff_factor=cf_cfg.gradient_cutoff_factor * 1.5,
            )
            if occl_pts is not None:
                front = _smooth_and_create_front(
                    occl_pts[0], occl_pts[1],
                    FrontType.OCCLUDED, center.id,
                    cf_cfg.spline_smoothing,
                )
                fronts.append(front)
                logger.info("Frente ocluido generado: %d puntos", front.npoints)

    return fronts


# --- Preparacion de campos ---


def _prepare_fields(center, ds, cfg):
    """Extrae theta_w, gradiente y viento alrededor del centro."""
    from mapa_frentes.fronts.tfp import _remove_time_dim, _ensure_2d, compute_theta_w

    ds = _remove_time_dim(ds)

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    try:
        theta_w = compute_theta_w(ds)
        theta_w = smooth_field(theta_w, sigma=cfg.tfp.smooth_sigma)
        gx, gy = spherical_gradient(theta_w, lats, lons)
        grad_mag = np.sqrt(gx**2 + gy**2)

        u850 = _ensure_2d(ds["u850"].values)
        v850 = _ensure_2d(ds["v850"].values)
    except Exception as e:
        logger.error("Error preparando campos: %s", e)
        return None, None, None, None, None, None, None, None

    return theta_w, gx, gy, grad_mag, u850, v850, lats, lons


# --- Scoring azimutal ---


def _azimuthal_advection_scores(
    center_lat: float,
    center_lon: float,
    theta_w: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    grad_mag: np.ndarray,
    u850: np.ndarray,
    v850: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    radius_deg: float,
    n_azimuths: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula scores de adveccion termica en un circulo alrededor del centro.

    Para cada azimut, interpola gradiente y viento, calcula la componente
    del viento perpendicular al gradiente * magnitud del gradiente.

    Returns:
        (azimuths, scores): arrays de longitud n_azimuths.
        score > 0 = adveccion fria (frente frio)
        score < 0 = adveccion calida (frente calido)
    """
    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    scores = np.zeros(n_azimuths)

    # Interpoladores
    interp_gx = RegularGridInterpolator(
        (lats, lons), gx, bounds_error=False, fill_value=0,
    )
    interp_gy = RegularGridInterpolator(
        (lats, lons), gy, bounds_error=False, fill_value=0,
    )
    interp_gm = RegularGridInterpolator(
        (lats, lons), grad_mag, bounds_error=False, fill_value=0,
    )
    interp_u = RegularGridInterpolator(
        (lats, lons), u850, bounds_error=False, fill_value=0,
    )
    interp_v = RegularGridInterpolator(
        (lats, lons), v850, bounds_error=False, fill_value=0,
    )

    for i, az in enumerate(azimuths):
        az_rad = np.radians(az)
        # Punto en el circulo (azimut desde N, sentido horario)
        dlat = radius_deg * np.cos(az_rad)
        dlon = radius_deg * np.sin(az_rad) / max(np.cos(np.radians(center_lat)), 0.1)
        pt_lat = center_lat + dlat
        pt_lon = center_lon + dlon
        pt = np.array([[pt_lat, pt_lon]])

        gx_val = interp_gx(pt).item()
        gy_val = interp_gy(pt).item()
        gm_val = interp_gm(pt).item()
        u_val = interp_u(pt).item()
        v_val = interp_v(pt).item()

        if gm_val < 1e-12:
            scores[i] = 0.0
            continue

        # Normal al gradiente (direccion perpendicular al frente)
        nx = gx_val / gm_val
        ny = gy_val / gm_val

        # Componente del viento en la direccion del gradiente
        vn = u_val * nx + v_val * ny

        # Score: viento cruzando el gradiente * magnitud del gradiente
        # Positivo = aire frio avanzando (frente frio)
        # Negativo = aire calido avanzando (frente calido)
        scores[i] = vn * gm_val

    return azimuths, scores


def _find_front_direction(
    azimuths: np.ndarray,
    scores: np.ndarray,
    sign: int,
) -> float | None:
    """Encuentra la direccion del frente frio (sign=+1) o calido (sign=-1).

    Busca el azimut con maximo |score| del signo pedido.
    Devuelve None si la senal es demasiado debil.
    """
    if sign > 0:
        mask = scores > 0
    else:
        mask = scores < 0

    if not np.any(mask):
        return None

    # Filtrar scores del signo pedido
    valid_scores = np.abs(scores[mask])
    valid_azimuths = azimuths[mask]

    # Umbral: al menos 20% del maximo score global
    global_max = np.max(np.abs(scores))
    if global_max < 1e-12:
        return None

    threshold = 0.2 * global_max
    strong = valid_scores >= threshold
    if not np.any(strong):
        return None

    # Azimut con maximo score
    best_idx = np.argmax(valid_scores[strong])
    return float(valid_azimuths[strong][best_idx])


# --- Ray-marching ---


def _trace_front_ray(
    start_lat: float,
    start_lon: float,
    initial_azimuth: float,
    grad_mag: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    max_length_deg: float = 15.0,
    step_deg: float = 0.5,
    max_turn_deg: float = 30.0,
    gradient_cutoff_factor: float = 0.3,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Traza un frente desde el centro hacia afuera siguiendo la cresta del gradiente.

    En cada paso:
    1. Avanzar step_deg en la direccion actual
    2. Evaluar el gradiente termico en la nueva posicion
    3. La cresta del gradiente (perpendicular al vector gradiente) sugiere
       la orientacion local del frente
    4. Ajustar la direccion hacia la cresta, limitado por max_turn_deg
    5. Parar si el gradiente cae demasiado o se sale del dominio

    Returns:
        (lats, lons) del frente trazado, o None si es demasiado corto.
    """
    interp_gm = RegularGridInterpolator(
        (lats, lons), grad_mag, bounds_error=False, fill_value=0,
    )
    interp_gx = RegularGridInterpolator(
        (lats, lons), gx, bounds_error=False, fill_value=0,
    )
    interp_gy = RegularGridInterpolator(
        (lats, lons), gy, bounds_error=False, fill_value=0,
    )

    # Gradiente en el punto de partida (para referencia de cutoff)
    start_gm = interp_gm(np.array([[start_lat, start_lon]])).item()
    if start_gm < 1e-12:
        # Buscar el maximo del gradiente a lo largo del primer paso
        az_rad = np.radians(initial_azimuth)
        test_lat = start_lat + step_deg * np.cos(az_rad)
        cos_lat = max(np.cos(np.radians(start_lat)), 0.1)
        test_lon = start_lon + step_deg * np.sin(az_rad) / cos_lat
        start_gm = interp_gm(np.array([[test_lat, test_lon]])).item()
    if start_gm < 1e-12:
        return None

    cutoff = gradient_cutoff_factor * start_gm
    peak_gm = start_gm

    ray_lats = [start_lat]
    ray_lons = [start_lon]
    total_length = 0.0

    # Direccion actual (azimut en grados desde N)
    current_az = initial_azimuth
    max_turn_rad = np.radians(max_turn_deg)

    max_steps = int(max_length_deg / step_deg) + 10

    for _ in range(max_steps):
        if total_length >= max_length_deg:
            break

        # Avanzar
        az_rad = np.radians(current_az)
        cos_lat = max(np.cos(np.radians(ray_lats[-1])), 0.1)
        new_lat = ray_lats[-1] + step_deg * np.cos(az_rad)
        new_lon = ray_lons[-1] + step_deg * np.sin(az_rad) / cos_lat

        # Verificar limites del dominio
        if (new_lat < lats.min() + 1 or new_lat > lats.max() - 1
                or new_lon < lons.min() + 1 or new_lon > lons.max() - 1):
            break

        # Evaluar gradiente
        pt = np.array([[new_lat, new_lon]])
        gm_val = interp_gm(pt).item()
        peak_gm = max(peak_gm, gm_val)
        cutoff = gradient_cutoff_factor * peak_gm

        if gm_val < cutoff:
            break

        ray_lats.append(new_lat)
        ray_lons.append(new_lon)
        total_length += step_deg

        # Ajustar direccion: la cresta del gradiente es perpendicular al vector gradiente
        gx_val = interp_gx(pt).item()
        gy_val = interp_gy(pt).item()
        gm_safe = max(gm_val, 1e-12)

        # Vector gradiente normalizado
        gnx = gx_val / gm_safe  # componente lon
        gny = gy_val / gm_safe  # componente lat

        # La cresta es perpendicular al gradiente -> dos opciones
        # Elegir la que esta mas cerca de la direccion actual
        ridge_az1 = np.degrees(np.arctan2(gny, -gnx))  # perpendicular opcion 1
        ridge_az2 = ridge_az1 + 180.0

        # Convertir a azimut desde N
        ridge_az1 = (90.0 - ridge_az1) % 360.0
        ridge_az2 = (ridge_az1 + 180.0) % 360.0

        # Elegir la mas cercana a current_az
        diff1 = _angle_diff(current_az, ridge_az1)
        diff2 = _angle_diff(current_az, ridge_az2)

        if abs(diff1) <= abs(diff2):
            target_az = ridge_az1
            diff = diff1
        else:
            target_az = ridge_az2
            diff = diff2

        # Limitar el giro
        if abs(diff) > max_turn_deg:
            diff = max_turn_deg * np.sign(diff)

        current_az = (current_az + diff) % 360.0

    # Minimo 4 puntos para un frente valido
    if len(ray_lats) < 4:
        return None

    return np.array(ray_lats), np.array(ray_lons)


def _angle_diff(az1: float, az2: float) -> float:
    """Diferencia angular en grados, resultado en [-180, 180]."""
    d = (az2 - az1) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


# --- Oclusion ---


def _occlusion_azimuth(
    cold_az: float,
    warm_az: float,
) -> float | None:
    """Calcula el azimut del frente ocluido (sector entre calido y frio por NW).

    El ocluido va desde el centro en la direccion opuesta al sector
    entre frio y calido (pasando por el lado "corto" del arco).
    """
    # Bisectriz del arco que va de warm_az a cold_az en sentido antihorario
    # (pasando por el sector N/NW)
    diff = _angle_diff(warm_az, cold_az)

    if abs(diff) < 60:
        # Frentes demasiado cercanos, no hay oclusion clara
        return None

    # Bisectriz en el lado opuesto al sector frio-calido
    mid = (warm_az + diff / 2.0) % 360.0
    # El ocluido va en la direccion opuesta a la bisectriz del sector abierto
    occl_az = (mid + 180.0) % 360.0

    return occl_az


# --- Suavizado y creacion ---


def _smooth_and_create_front(
    lats_raw: np.ndarray,
    lons_raw: np.ndarray,
    front_type: FrontType,
    center_id: str,
    smoothing: float,
) -> Front:
    """Suaviza la polylinea con spline y crea un objeto Front."""
    from mapa_frentes.fronts.connector import _smooth_spline

    if len(lats_raw) >= 4:
        sm_lats, sm_lons = _smooth_spline(lats_raw, lons_raw, smoothing=smoothing)
    else:
        sm_lats, sm_lons = lats_raw, lons_raw

    return Front(
        front_type=front_type,
        lats=sm_lats,
        lons=sm_lons,
        id=f"gen_{center_id}_{front_type.value}",
        center_id=center_id,
    )
