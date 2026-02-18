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

EARTH_RADIUS_KM = 6371.0


# -----------------------------------------------------------------------------
# Distancias robustas (haversine) + helpers
# -----------------------------------------------------------------------------

def _wrap_lon(lon: float) -> float:
    """Normaliza longitudes a [-180, 180)."""
    x = (lon + 180.0) % 360.0 - 180.0
    # evita -180 exacto si quieres consistencia
    return float(x)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Distancia haversine (km). Soporta arrays.
    lat/lon en grados.
    """
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # wrap del delta lon a [-pi, pi]
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def _deg_to_km_approx(deg: float, lat_ref: float = 45.0) -> float:
    """
    Convierte "grados" a km de forma aproximada usando lat_ref para lon.
    Útil para convertir umbrales del config (en grados) a km.
    """
    # 1 deg lat ~ 111 km
    km_lat = 111.0 * deg
    km_lon = 111.0 * np.cos(np.deg2rad(lat_ref)) * deg
    # usa algo intermedio
    return float(0.5 * (km_lat + km_lon))


def _front_min_distance_km(front: Front, center: PressureCenter) -> float:
    """Distancia mínima (km) desde cualquier punto del frente al centro."""
    lats = front.lats
    lons = np.array([_wrap_lon(x) for x in front.lons], dtype=float)
    c_lat = center.lat
    c_lon = _wrap_lon(center.lon)

    d = _haversine_km(lats, lons, c_lat, c_lon)
    return float(np.nanmin(d)) if d.size else float("inf")


def _front_end_distances_km(front: Front, center: PressureCenter) -> tuple[float, float]:
    """Distancias (km) centro->extremo start y centro->extremo end."""
    c_lat = center.lat
    c_lon = _wrap_lon(center.lon)
    d_start = float(_haversine_km(front.lats[0], _wrap_lon(front.lons[0]), c_lat, c_lon))
    d_end = float(_haversine_km(front.lats[-1], _wrap_lon(front.lons[-1]), c_lat, c_lon))
    return d_start, d_end


# -----------------------------------------------------------------------------
# Asociacion
# -----------------------------------------------------------------------------

def find_nearest_center_for_front(
    front: Front,
    lows: list[PressureCenter],
    *,
    strategy: str = "min_along_front",   # "ends" o "min_along_front"
) -> tuple[PressureCenter | None, str | None, float]:
    """Encuentra el centro L mas cercano y el extremo a asociar.

    strategy:
      - "ends": solo compara distancias a start/end (legacy)
      - "min_along_front": elige el centro por distancia mínima a cualquier punto;
        el extremo asociado se decide por cuál extremo queda más cerca del centro.

    Returns:
        (center, which_end, distance_km)
    """
    best_dist = float("inf")
    best_center = None
    best_end = None

    for center in lows:
        if strategy == "ends":
            d_start, d_end = _front_end_distances_km(front, center)
            if d_start < d_end and d_start < best_dist:
                best_dist, best_center, best_end = d_start, center, "start"
            elif d_end <= d_start and d_end < best_dist:
                best_dist, best_center, best_end = d_end, center, "end"
        else:
            # robusto: decide centro por mínima distancia a lo largo del frente
            d_min = _front_min_distance_km(front, center)
            if d_min < best_dist:
                # qué extremo es "mejor" para extender
                d_start, d_end = _front_end_distances_km(front, center)
                which = "start" if d_start < d_end else "end"
                best_dist, best_center, best_end = d_min, center, which

    return best_center, best_end, best_dist


def associate_fronts_to_centers(
    collection: FrontCollection,
    centers: list[PressureCenter],
    cfg: AppConfig,
) -> FrontCollection:
    """Asocia frentes existentes al centro L mas cercano (solo metadata o extend opcional).

    Para cada frente:
      1) Busca mejor centro L con estrategia robusta
      2) Si distancia <= umbral, asigna center_id y association_end
      3) Opcional: extender geométricamente hacia el centro (Bezier + spline)
    """
    lows = [c for c in centers if c.type == "L"]
    if not lows:
        return collection

    # Umbral en km (partimos del config en grados, para no romper tu YAML)
    max_dist_deg = float(cfg.center_fronts.max_association_distance_deg)
    lat_ref = float(getattr(cfg.plotting.projection_params, "central_latitude", 45.0)) if hasattr(cfg, "plotting") else 45.0
    max_dist_km = _deg_to_km_approx(max_dist_deg, lat_ref=lat_ref)

    # Nuevos flags (opcionales)
    strategy = getattr(cfg.center_fronts, "association_strategy", "min_along_front")
    do_extend = bool(getattr(cfg.center_fronts, "extend_to_center", False))
    extend_only_if_close_km = float(getattr(cfg.center_fronts, "extend_max_distance_km", max_dist_km))

    associated = 0
    extended = 0

    for front in collection:
        if front.front_type == FrontType.INSTABILITY_LINE:
            continue
        if front.center_id:
            continue

        best_center, best_end, best_dist_km = find_nearest_center_for_front(
            front, lows, strategy=strategy
        )

        if best_center is None or best_end is None:
            continue

        if best_dist_km <= max_dist_km:
            front.center_id = best_center.id
            front.association_end = best_end
            associated += 1

            # Extensión geométrica opcional (esto ayuda a que “nazca” del centro como en análisis manual)
            if do_extend and best_dist_km <= extend_only_if_close_km:
                try:
                    smooth_extend_to_center(front, best_center, best_end, cfg)
                    extended += 1
                except Exception as e:
                    logger.warning("No se pudo extender frente %s a centro %s: %s", front.id, best_center.id, e)

    logger.info(
        "Asociacion: %d frentes conectados a centros L de %d totales (extendidos: %d)",
        associated, len(collection), extended,
    )
    return collection


def filter_fronts_near_lows(
    collection: FrontCollection,
    centers: list[PressureCenter],
    cfg: AppConfig,
) -> FrontCollection:
    """Filtra frentes que no estén cerca de ningún centro de baja presión.

    Mantiene:
      - frentes ya asociados (center_id)
      - lineas de inestabilidad

    Para el resto: elimina si la distancia mínima a cualquier L es > umbral.
    """
    if not cfg.center_fronts.require_low_center:
        return collection

    lows = [c for c in centers if c.type == "L"]
    if not lows:
        logger.warning("No hay centros L: se mantienen todos los frentes")
        return collection

    max_dist_deg = float(cfg.center_fronts.max_distance_to_low_deg)
    lat_ref = float(getattr(cfg.plotting.projection_params, "central_latitude", 45.0)) if hasattr(cfg, "plotting") else 45.0
    max_dist_km = _deg_to_km_approx(max_dist_deg, lat_ref=lat_ref)

    to_remove: list[str] = []

    for front in collection:
        if front.center_id:
            continue
        if front.front_type == FrontType.INSTABILITY_LINE:
            continue

        # distancia mínima real (km) a cualquier baja
        min_km = float("inf")
        for center in lows:
            d = _front_min_distance_km(front, center)
            if d < min_km:
                min_km = d

        if min_km > max_dist_km:
            to_remove.append(front.id)

    for fid in to_remove:
        collection.remove(fid)

    if to_remove:
        logger.info(
            "Filtro por proximidad a L: eliminados %d frentes lejanos, quedan %d",
            len(to_remove), len(collection),
        )

    return collection


# -----------------------------------------------------------------------------
# Extensión geométrica (igual que tu versión, con pequeños ajustes robustos)
# -----------------------------------------------------------------------------

def smooth_extend_to_center(
    front: Front,
    center: PressureCenter,
    which_end: str,
    cfg: AppConfig,
):
    """Extiende un frente hasta un centro con curva suave tangente.

    Modifica el frente in-place (concatena puntos).
    """
    from mapa_frentes.fronts.connector import _smooth_spline

    c_lat, c_lon = center.lat, _wrap_lon(center.lon)

    if which_end == "start":
        p0_lat, p0_lon = float(front.lats[0]), _wrap_lon(float(front.lons[0]))
        if front.npoints >= 3:
            p1_lat, p1_lon = float(front.lats[2]), _wrap_lon(float(front.lons[2]))
        else:
            p1_lat, p1_lon = float(front.lats[1]), _wrap_lon(float(front.lons[1]))
    else:
        p0_lat, p0_lon = float(front.lats[-1]), _wrap_lon(float(front.lons[-1]))
        if front.npoints >= 3:
            p1_lat, p1_lon = float(front.lats[-3]), _wrap_lon(float(front.lons[-3]))
        else:
            p1_lat, p1_lon = float(front.lats[-2]), _wrap_lon(float(front.lons[-2]))

    dist_km = float(_haversine_km(p0_lat, p0_lon, c_lat, c_lon))
    if dist_km < 10.0:  # si ya está prácticamente encima
        front.center_id = center.id
        front.association_end = which_end
        return

    # Tangente en grados (solo para dirección, no magnitud)
    tan_lat = p0_lat - p1_lat
    tan_lon = _wrap_lon(p0_lon - p1_lon)
    tan_norm = np.sqrt(tan_lat**2 + tan_lon**2)
    if tan_norm > 1e-10:
        tan_lat /= tan_norm
        tan_lon /= tan_norm

    # Distancia en "grados equivalentes" para el control (suave)
    # convertimos km a ~deg a lat media del segmento
    lat_ref = 0.5 * (p0_lat + c_lat)
    km_per_deg = 111.0
    dist_deg = dist_km / km_per_deg

    # Puntos de control Bezier
    cp1_lat = p0_lat + tan_lat * dist_deg / 3.0
    cp1_lon = _wrap_lon(p0_lon + tan_lon * dist_deg / 3.0)

    cp2_lat = c_lat + (p0_lat - c_lat) / 3.0
    cp2_lon = _wrap_lon(c_lon + (_wrap_lon(p0_lon - c_lon)) / 3.0)

    # Evaluar Bezier
    n_pts = max(int(dist_deg / 0.25), 10)
    t = np.linspace(0.0, 1.0, n_pts)
    b0 = (1 - t) ** 3
    b1 = 3 * t * (1 - t) ** 2
    b2 = 3 * t ** 2 * (1 - t)
    b3 = t ** 3

    ext_lats = b0 * p0_lat + b1 * cp1_lat + b2 * cp2_lat + b3 * c_lat
    ext_lons = b0 * p0_lon + b1 * cp1_lon + b2 * cp2_lon + b3 * c_lon
    ext_lons = np.array([_wrap_lon(x) for x in ext_lons], dtype=float)

    # Suavizado spline
    if len(ext_lats) >= 4:
        ext_lats, ext_lons = _smooth_spline(ext_lats, ext_lons, smoothing=0.3)

    # Excluir primer punto (ya existe)
    ext_lats = ext_lats[1:]
    ext_lons = ext_lons[1:]

    if which_end == "start":
        front.lats = np.concatenate([ext_lats[::-1], front.lats])
        front.lons = np.concatenate([ext_lons[::-1], front.lons])
    else:
        front.lats = np.concatenate([front.lats, ext_lats])
        front.lons = np.concatenate([front.lons, ext_lons])

    front.center_id = center.id
    front.association_end = which_end


def create_occlusion_from_pair(
    collection: FrontCollection,
    centers: list[PressureCenter],
    cfg: AppConfig,
) -> FrontCollection:
    """Prolonga frentes frio/calido por la cresta de |nabla theta_w| y busca cruce.

    Algoritmo:
    1. Para cada centro L, busca un COLD y un WARM cercanos
    2. Prolonga cada frente desde su endpoint cercano al centro,
       siguiendo la cresta de |nabla theta_w| (dir. perpendicular a nabla|nabla theta_w|)
    3. Si las prolongaciones se cruzan: ese es el triple point
       - Extiende cold y warm hasta el triple point
       - Crea frente OCCLUDED desde centro L hasta triple point
    4. Si NO se cruzan: descarta las prolongaciones (deja frentes intactos)
    """
    from scipy.interpolate import RegularGridInterpolator
    from mapa_frentes.fronts.connector import _smooth_spline

    lows = [c for c in centers if c.type == "L"]
    created = 0

    # Campos necesarios del metadata
    grad_mag = collection.metadata.get("grad_mag")
    gmag_x = collection.metadata.get("gmag_x")
    gmag_y = collection.metadata.get("gmag_y")
    meta_lats = collection.metadata.get("lats")
    meta_lons = collection.metadata.get("lons")

    if any(v is None for v in (grad_mag, gmag_x, gmag_y, meta_lats, meta_lons)):
        logger.warning("Faltan campos en metadata para prolongar frentes")
        return collection

    interp_grad = RegularGridInterpolator(
        (meta_lats, meta_lons), grad_mag,
        bounds_error=False, fill_value=0.0,
    )
    interp_gmx = RegularGridInterpolator(
        (meta_lats, meta_lons), gmag_x,
        bounds_error=False, fill_value=0.0,
    )
    interp_gmy = RegularGridInterpolator(
        (meta_lats, meta_lons), gmag_y,
        bounds_error=False, fill_value=0.0,
    )

    for center in lows:
        # Buscar frentes frios y calidos cerca de este centro
        cold_fronts = []
        warm_fronts = []
        max_km = _deg_to_km_approx(
            float(cfg.center_fronts.max_distance_to_low_deg)
        )
        for f in collection.fronts:
            if f.front_type == FrontType.COLD:
                if f.center_id == center.id:
                    cold_fronts.append(f)
                elif not f.center_id and _front_min_distance_km(f, center) <= max_km:
                    cold_fronts.append(f)
            elif f.front_type == FrontType.WARM:
                if f.center_id == center.id:
                    warm_fronts.append(f)
                elif not f.center_id and _front_min_distance_km(f, center) <= max_km:
                    warm_fronts.append(f)

        if not cold_fronts or not warm_fronts:
            continue

        cold = max(cold_fronts, key=lambda f: f.npoints)
        warm = max(warm_fronts, key=lambda f: f.npoints)

        cold_near = _nearest_end(cold, center)
        warm_near = _nearest_end(warm, center)

        # Tangente del frente en el endpoint (direccion de prolongacion)
        cold_dir = _front_tangent_at_end(cold, cold_near)
        warm_dir = _front_tangent_at_end(warm, warm_near)

        cold_ep = _get_endpoint(cold, cold_near)
        warm_ep = _get_endpoint(warm, warm_near)

        # Prolongar cada frente por la cresta de |nabla theta_w|
        cold_ext = _extend_along_ridge(
            cold_ep[0], cold_ep[1],
            cold_dir[0], cold_dir[1],
            interp_grad, interp_gmx, interp_gmy,
            step_km=25.0, max_steps=100, coast_steps=15,
        )
        warm_ext = _extend_along_ridge(
            warm_ep[0], warm_ep[1],
            warm_dir[0], warm_dir[1],
            interp_grad, interp_gmx, interp_gmy,
            step_km=25.0, max_steps=100, coast_steps=15,
        )

        if cold_ext is None or warm_ext is None:
            logger.debug("No se pudo prolongar frio/calido en centro %s", center.id)
            continue

        # Buscar cruce entre las prolongaciones
        tp = _find_intersection(cold_ext, warm_ext, tolerance_deg=1.5)

        logger.info(
            "Centro %s: cold_ext %d pts (%.1f,%.1f)->(%.1f,%.1f), "
            "warm_ext %d pts (%.1f,%.1f)->(%.1f,%.1f), cruce=%s",
            center.id,
            len(cold_ext), cold_ext[0, 0], cold_ext[0, 1],
            cold_ext[-1, 0], cold_ext[-1, 1],
            len(warm_ext), warm_ext[0, 0], warm_ext[0, 1],
            warm_ext[-1, 0], warm_ext[-1, 1],
            tp is not None,
        )

        if tp is None:
            # Log distancia minima para diagnostico
            dlat = cold_ext[:, 0:1] - warm_ext[:, 0:1].T
            dlon = cold_ext[:, 1:2] - warm_ext[:, 1:2].T
            min_d = float(np.sqrt(dlat**2 + dlon**2).min())
            logger.info(
                "Prolongaciones no se cruzan en centro %s: dist minima=%.2f deg",
                center.id, min_d,
            )
            continue

        tp_lat, tp_lon, cold_idx, warm_idx = tp

        # --- Existen cruce: extender frentes hasta el triple point ---
        cold_ext_pts = cold_ext[:cold_idx + 1]
        warm_ext_pts = warm_ext[:warm_idx + 1]

        # Añadir la prolongacion al frente (excluyendo el primer punto que ya existe)
        if len(cold_ext_pts) > 1:
            ext_lats = np.append(cold_ext_pts[1:, 0], tp_lat)
            ext_lons = np.append(cold_ext_pts[1:, 1], tp_lon)
            if cold_near == "start":
                cold.lats = np.concatenate([ext_lats[::-1], cold.lats])
                cold.lons = np.concatenate([ext_lons[::-1], cold.lons])
            else:
                cold.lats = np.concatenate([cold.lats, ext_lats])
                cold.lons = np.concatenate([cold.lons, ext_lons])

        if len(warm_ext_pts) > 1:
            ext_lats = np.append(warm_ext_pts[1:, 0], tp_lat)
            ext_lons = np.append(warm_ext_pts[1:, 1], tp_lon)
            if warm_near == "start":
                warm.lats = np.concatenate([ext_lats[::-1], warm.lats])
                warm.lons = np.concatenate([ext_lons[::-1], warm.lons])
            else:
                warm.lats = np.concatenate([warm.lats, ext_lats])
                warm.lons = np.concatenate([warm.lons, ext_lons])

        # --- Crear frente ocluido: centro L → triple point ---
        # Trazar por la cresta tambien
        occl_dir_lat = tp_lat - center.lat
        occl_dir_lon = tp_lon - center.lon
        occl_ext = _extend_along_ridge(
            center.lat, center.lon,
            occl_dir_lat, occl_dir_lon,
            interp_grad, interp_gmx, interp_gmy,
            step_km=20.0, max_steps=80, coast_steps=10,
        )

        dist_to_tp = np.sqrt(
            (center.lat - tp_lat)**2 + (center.lon - tp_lon)**2
        )
        if dist_to_tp < 0.3:
            continue

        # Usar la prolongacion desde el centro si llega cerca del TP,
        # si no, linea recta
        if occl_ext is not None and len(occl_ext) >= 2:
            # Recortar hasta el triple point
            dists_to_tp = np.sqrt(
                (occl_ext[:, 0] - tp_lat)**2 + (occl_ext[:, 1] - tp_lon)**2
            )
            nearest_idx = int(np.argmin(dists_to_tp))
            occl_lats = occl_ext[:nearest_idx + 1, 0]
            occl_lons = occl_ext[:nearest_idx + 1, 1]
            # Asegurar que termina en TP
            occl_lats = np.append(occl_lats, tp_lat)
            occl_lons = np.append(occl_lons, tp_lon)
        else:
            n_pts = max(int(dist_to_tp / 0.25), 4)
            t = np.linspace(0.0, 1.0, n_pts)
            occl_lats = center.lat + t * (tp_lat - center.lat)
            occl_lons = center.lon + t * (tp_lon - center.lon)

        # Suavizar
        if len(occl_lats) >= 4:
            try:
                occl_lats, occl_lons = _smooth_spline(
                    occl_lats, occl_lons, smoothing=0.3,
                )
            except Exception:
                pass

        if len(occl_lats) < 2:
            continue

        occl_front = Front(
            front_type=FrontType.OCCLUDED,
            lats=occl_lats,
            lons=occl_lons,
            id=f"occl_{center.id}",
            center_id=center.id,
        )
        collection.add(occl_front)
        created += 1
        logger.info(
            "Ocluido creado para centro %s: triple point (%.1f, %.1f), "
            "cold ext %d pts, warm ext %d pts, occl %d pts",
            center.id, tp_lat, tp_lon,
            len(cold_ext_pts), len(warm_ext_pts), occl_front.npoints,
        )

    if created:
        logger.info("Ocluidos creados: %d", created)
    return collection


# -----------------------------------------------------------------------------
# Prolongacion por cresta de |nabla theta_w|
# -----------------------------------------------------------------------------

def _extend_along_ridge(
    start_lat: float,
    start_lon: float,
    dir_lat: float,
    dir_lon: float,
    interp_grad,
    interp_gmx,
    interp_gmy,
    step_km: float = 25.0,
    max_steps: int = 100,
    grad_cutoff_factor: float = 0.05,
    direction_inertia: float = 0.6,
    coast_steps: int = 15,
) -> np.ndarray | None:
    """Prolonga un frente siguiendo la cresta de |nabla theta_w|.

    La cresta es perpendicular a nabla|nabla theta_w| en espacio fisico.

    IMPORTANTE: gmag_x = componente ZONAL (E-W, unidades/m),
                gmag_y = componente MERIDIONAL (N-S, unidades/m).
    La perpendicular en espacio fisico a (gx_phys, gy_phys) es (-gy_phys, gx_phys).
    Para avanzar en grados: dlat = dy_m / R, dlon = dx_m / (R * cos(lat)).

    En cada paso:
    1. Calcula nabla|nabla theta_w| en espacio fisico = (gx_zonal, gy_merid)
    2. Direccion de la cresta (perpendicular) en espacio fisico
    3. Convierte a incrementos (dlat, dlon) usando la metrica esferica
    4. Avanza con inercia para suavizar curvas
    5. Para si |nabla theta_w| cae por debajo de un umbral durante
       coast_steps pasos consecutivos

    Returns:
        Array (N, 2) de [lat, lon] o None si falla.
    """
    R_EARTH = 6371000.0  # metros

    # Convertir direccion inicial (en grados lat/lon) a espacio fisico (metros)
    cos_lat0 = np.cos(np.radians(start_lat))
    dx_phys = dir_lon * np.radians(1.0) * R_EARTH * cos_lat0
    dy_phys = dir_lat * np.radians(1.0) * R_EARTH
    d_norm = np.sqrt(dx_phys**2 + dy_phys**2)
    if d_norm < 1e-6:
        return None
    # Direccion unitaria en espacio fisico (metros)
    vx = dx_phys / d_norm  # componente zonal
    vy = dy_phys / d_norm  # componente meridional

    cur_lat, cur_lon = start_lat, start_lon
    path = [[cur_lat, cur_lon]]

    # Umbral: parar cuando el gradiente cae mucho y no se recupera
    initial_grad = float(interp_grad([[cur_lat, cur_lon]])[0])
    grad_threshold = initial_grad * grad_cutoff_factor
    if initial_grad < 1e-12:
        return None

    steps_below = 0

    for step_i in range(max_steps):
        pt = [[cur_lat, cur_lon]]
        # gmag_x = componente zonal de nabla|nabla theta_w| (unidades/m²)
        # gmag_y = componente meridional de nabla|nabla theta_w| (unidades/m²)
        gx_zon = float(interp_gmx(pt)[0])
        gy_mer = float(interp_gmy(pt)[0])

        gm = np.sqrt(gx_zon**2 + gy_mer**2)
        if gm < 1e-20:
            # Plateau: avanzar recto (mantener vx, vy)
            ridge_x, ridge_y = vx, vy
        else:
            # Perpendicular a nabla|nabla theta_w| en espacio fisico
            # nabla = (gx_zon, gy_mer), perpendicular = (-gy_mer, gx_zon)
            r1_x, r1_y = -gy_mer / gm, gx_zon / gm
            r2_x, r2_y = gy_mer / gm, -gx_zon / gm

            # Elegir la alineada con la direccion de avance
            dot1 = r1_x * vx + r1_y * vy
            dot2 = r2_x * vx + r2_y * vy

            if dot1 >= dot2:
                ridge_x, ridge_y = r1_x, r1_y
            else:
                ridge_x, ridge_y = r2_x, r2_y

        # Mezclar con inercia (suaviza curvas) - en espacio fisico
        new_vx = direction_inertia * vx + (1 - direction_inertia) * ridge_x
        new_vy = direction_inertia * vy + (1 - direction_inertia) * ridge_y
        d_norm = np.sqrt(new_vx**2 + new_vy**2)
        if d_norm < 1e-10:
            break
        vx = new_vx / d_norm
        vy = new_vy / d_norm

        # Convertir paso de km a grados lat/lon
        step_m = step_km * 1000.0
        cos_lat = np.cos(np.radians(cur_lat))
        if cos_lat < 0.01:
            break  # cerca del polo
        dlat_deg = np.degrees(vy * step_m / R_EARTH)
        dlon_deg = np.degrees(vx * step_m / (R_EARTH * cos_lat))

        cur_lat += dlat_deg
        cur_lon += dlon_deg

        # Comprobar si el gradiente sigue siendo significativo
        g = float(interp_grad([[cur_lat, cur_lon]])[0])
        if g < grad_threshold:
            steps_below += 1
            if steps_below >= coast_steps:
                logger.debug(
                    "Ridge extension: grad bajo %.2e < %.2e durante %d pasos, "
                    "parando en (%.2f, %.2f) tras %d pasos",
                    g, grad_threshold, coast_steps, cur_lat, cur_lon, step_i,
                )
                break
        else:
            steps_below = 0

        path.append([cur_lat, cur_lon])

    result = np.array(path)
    logger.info(
        "Ridge extension: %d puntos desde (%.2f, %.2f) hasta (%.2f, %.2f), "
        "grad_inicial=%.2e, umbral=%.2e, steps_below=%d",
        len(result), start_lat, start_lon,
        result[-1, 0], result[-1, 1],
        initial_grad, grad_threshold, steps_below,
    )
    return result if len(result) >= 2 else None


def _find_intersection(
    path_a: np.ndarray,
    path_b: np.ndarray,
    tolerance_deg: float = 0.5,
) -> tuple[float, float, int, int] | None:
    """Busca el punto donde dos caminos se cruzan (dentro de tolerancia).

    Returns:
        (lat, lon, idx_a, idx_b) del punto de cruce mas cercano,
        o None si no se cruzan.
    """
    if len(path_a) < 2 or len(path_b) < 2:
        return None

    # Matriz de distancias entre todos los pares de puntos
    dlat = path_a[:, 0:1] - path_b[:, 0:1].T  # (Na, Nb)
    dlon = path_a[:, 1:2] - path_b[:, 1:2].T
    dists = np.sqrt(dlat**2 + dlon**2)

    min_dist = dists.min()
    if min_dist > tolerance_deg:
        return None

    # Encontrar el par mas cercano
    idx_a, idx_b = np.unravel_index(dists.argmin(), dists.shape)

    # Triple point = punto medio entre los dos puntos mas cercanos
    tp_lat = 0.5 * (path_a[idx_a, 0] + path_b[idx_b, 0])
    tp_lon = 0.5 * (path_a[idx_a, 1] + path_b[idx_b, 1])

    return (float(tp_lat), float(tp_lon), int(idx_a), int(idx_b))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _nearest_end(front: Front, center: PressureCenter) -> str:
    """Devuelve 'start' o 'end' segun que extremo esta mas cerca del centro."""
    d_start = float(_haversine_km(
        front.lats[0], front.lons[0], center.lat, center.lon,
    ))
    d_end = float(_haversine_km(
        front.lats[-1], front.lons[-1], center.lat, center.lon,
    ))
    return "start" if d_start < d_end else "end"


def _get_endpoint(front: Front, which: str) -> tuple[float, float]:
    """Devuelve (lat, lon) del extremo indicado."""
    if which == "start":
        return float(front.lats[0]), float(front.lons[0])
    return float(front.lats[-1]), float(front.lons[-1])


def _front_tangent_at_end(front: Front, which_end: str) -> tuple[float, float]:
    """Calcula la tangente del frente en el extremo indicado (direccion de salida).

    Usa los ultimos 3-5 puntos para calcular la direccion.
    El vector apunta HACIA FUERA del frente (para prolongar).
    """
    n = min(5, front.npoints)
    if which_end == "start":
        # Tangente en start: apunta desde interior hacia start (para prolongar)
        ref_lat = float(np.mean(front.lats[1:n]))
        ref_lon = float(np.mean(front.lons[1:n]))
        ep_lat = float(front.lats[0])
        ep_lon = float(front.lons[0])
    else:
        # Tangente en end: apunta desde interior hacia end
        ref_lat = float(np.mean(front.lats[-n:-1]))
        ref_lon = float(np.mean(front.lons[-n:-1]))
        ep_lat = float(front.lats[-1])
        ep_lon = float(front.lons[-1])

    return (ep_lat - ref_lat, ep_lon - ref_lon)


# -----------------------------------------------------------------------------
# Filtro de giros bruscos
# -----------------------------------------------------------------------------

def trim_sharp_turns(
    collection: FrontCollection,
    max_turn_deg: float = 90.0,
    min_points: int = 5,
) -> FrontCollection:
    """Recorta frentes con giros bruscos: corta en el punto de maxima
    curvatura y se queda con el tramo mas largo.

    Para cada frente, calcula el angulo de giro en cada vertice interior.
    Si alguno supera max_turn_deg, corta ahi y conserva el segmento mas largo.
    Repite hasta que no haya giros bruscos.

    Args:
        collection: Coleccion de frentes.
        max_turn_deg: Angulo maximo permitido entre segmentos consecutivos.
        min_points: Minimo de puntos para que un tramo sea valido.
    """
    R_EARTH = 6371000.0
    trimmed = 0
    new_fronts = []

    for front in collection.fronts:
        lats, lons = front.lats.copy(), front.lons.copy()
        changed = True

        while changed and len(lats) >= min_points:
            changed = False
            if len(lats) < 3:
                break

            # Calcular vectores entre puntos consecutivos en espacio fisico
            cos_lat = np.cos(np.radians(lats))
            dx = np.diff(lons) * np.radians(1.0) * R_EARTH * cos_lat[:-1]
            dy = np.diff(lats) * np.radians(1.0) * R_EARTH

            # Angulos de giro en cada vertice interior
            worst_angle = 0.0
            worst_idx = -1
            for k in range(len(dx) - 1):
                # Vectores consecutivos
                ax, ay = dx[k], dy[k]
                bx, by = dx[k + 1], dy[k + 1]
                dot = ax * bx + ay * by
                mag_a = np.sqrt(ax**2 + ay**2)
                mag_b = np.sqrt(bx**2 + by**2)
                if mag_a < 1e-6 or mag_b < 1e-6:
                    continue
                cos_angle = np.clip(dot / (mag_a * mag_b), -1.0, 1.0)
                turn_deg = np.degrees(np.arccos(cos_angle))
                if turn_deg > worst_angle:
                    worst_angle = turn_deg
                    worst_idx = k + 1  # indice del vertice con el giro

            if worst_angle > max_turn_deg and worst_idx > 0:
                # Cortar: quedarse con el tramo mas largo
                seg_a_len = worst_idx
                seg_b_len = len(lats) - worst_idx
                if seg_a_len >= seg_b_len:
                    lats = lats[:worst_idx]
                    lons = lons[:worst_idx]
                else:
                    lats = lats[worst_idx:]
                    lons = lons[worst_idx:]
                changed = True
                trimmed += 1

        if len(lats) >= min_points:
            front.lats = lats
            front.lons = lons
            new_fronts.append(front)
        else:
            trimmed += 1

    if trimmed:
        logger.info("trim_sharp_turns: %d cortes/eliminaciones (umbral %.0f°)",
                     trimmed, max_turn_deg)

    collection.fronts = new_fronts
    return collection
