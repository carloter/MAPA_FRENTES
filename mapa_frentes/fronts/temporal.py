"""Seguimiento temporal de frentes para filtrado por persistencia y clasificacion.

Descarga varios timesteps (context_steps alrededor del central), detecta
frentes en cada uno, empareja frentes entre timesteps consecutivos y:
1. Filtra por persistencia: solo conserva frentes que aparecen en al menos
   min_persistence timesteps adicionales.
2. Clasifica por movimiento: la direccion de desplazamiento entre timesteps
   distingue frentes frios (avanzan hacia aire calido) de calidos.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import KDTree

from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType
from mapa_frentes.fronts.tfp import compute_tfp_fronts, _remove_time_dim, _ensure_2d, compute_theta_w
from mapa_frentes.fronts.classifier import classify_fronts
from mapa_frentes.fronts.instability import detect_instability_lines
from mapa_frentes.utils.geo import spherical_gradient
from mapa_frentes.utils.smoothing import smooth_field

logger = logging.getLogger(__name__)


def compute_temporal_fronts(
    cfg: AppConfig,
    date: datetime | None = None,
    base_step: int = 0,
    no_download: bool = False,
) -> FrontCollection:
    """Pipeline completo temporal: multi-step, matching, persistencia, clasificacion.

    Args:
        cfg: Configuracion de la aplicacion.
        date: Fecha/hora del run. None = ultimo disponible.
        base_step: Paso de prediccion central en horas.
        no_download: Si True, usa solo datos existentes en cache.

    Returns:
        FrontCollection con los frentes filtrados y clasificados del timestep central.
    """
    from mapa_frentes.data.ecmwf_download import download_ecmwf
    from mapa_frentes.data.grib_reader import read_grib_files

    temp_cfg = cfg.temporal
    offsets = sorted(temp_cfg.context_steps)

    # Indice del timestep central (offset 0) en la lista ordenada
    if 0 not in offsets:
        offsets.append(0)
        offsets.sort()
    center_idx = offsets.index(0)

    # 1. Descargar y detectar frentes en cada timestep
    all_fronts: list[FrontCollection] = []
    ds_center: xr.Dataset | None = None

    for i, offset in enumerate(offsets):
        step = base_step + offset
        if step < 0:
            logger.info("Saltando step %d (negativo)", step)
            all_fronts.append(FrontCollection())
            continue

        logger.info(
            "Temporal [%d/%d]: procesando step %d (offset %+d)...",
            i + 1, len(offsets), step, offset,
        )

        try:
            if no_download:
                grib_paths = _find_cached_gribs(cfg, date, step)
                if grib_paths is None:
                    logger.warning("No hay cache para step %d, saltando", step)
                    all_fronts.append(FrontCollection())
                    continue
            else:
                grib_paths = download_ecmwf(cfg, date=date, step=step)

            ds = read_grib_files(grib_paths, cfg)
            fronts = compute_tfp_fronts(ds, cfg)
            logger.info("Step %d: %d frentes detectados", step, len(fronts))

            # Guardar dataset del timestep central para clasificacion
            if offset == 0:
                ds_center = ds
            else:
                del ds

            all_fronts.append(fronts)

        except Exception as e:
            logger.warning("Error procesando step %d: %s", step, e)
            all_fronts.append(FrontCollection())

    if ds_center is None:
        logger.error("No se pudo procesar el timestep central (step %d)", base_step)
        return FrontCollection()

    # 2. Matching entre timesteps consecutivos
    matches = _match_consecutive(all_fronts, temp_cfg.match_distance_deg)

    # 3. Filtrar por persistencia
    center_fronts = all_fronts[center_idx]
    persistent = _filter_by_persistence(
        all_fronts, matches, center_idx, temp_cfg.min_persistence,
    )
    logger.info(
        "Persistencia: %d -> %d frentes",
        len(center_fronts), len(persistent),
    )

    # 4. Clasificar por movimiento
    _classify_persistent_fronts(
        persistent, all_fronts, matches, center_idx, ds_center, cfg,
    )

    # 5. Lineas de inestabilidad (solo del timestep central)
    instab_lines = detect_instability_lines(ds_center, cfg)
    for il in instab_lines:
        persistent.add(il)

    del ds_center
    return persistent


def _find_cached_gribs(
    cfg: AppConfig, date: datetime | None, step: int,
) -> dict[str, Path] | None:
    """Busca GRIBs cacheados para un step dado."""
    cache_dir = Path(cfg.data.cache_dir)

    if date is not None:
        date_tag = date.strftime("%Y%m%d%H")
        step_tag = f"_T{step:03d}" if step > 0 else ""
        sfc = cache_dir / f"ecmwf_sfc_{date_tag}{step_tag}.grib2"
        pl = cache_dir / f"ecmwf_pl850_{date_tag}{step_tag}.grib2"
    else:
        step_tag = f"_T{step:03d}" if step > 0 else ""
        sfc = cache_dir / f"ecmwf_sfc_latest{step_tag}.grib2"
        pl = cache_dir / f"ecmwf_pl850_latest{step_tag}.grib2"

    if sfc.exists() and pl.exists():
        return {"surface": sfc, "pressure": pl}
    return None


# --- Matching ---


def _mean_min_distance(front_a: Front, front_b: Front) -> float:
    """Distancia media de minimos punto a punto (simetrica).

    d(A,B) = 0.5 * (mean_i(min_j(dist(Ai,Bj))) + mean_j(min_i(dist(Ai,Bj))))
    """
    coords_a = front_a.coords  # (N, 2) [lon, lat]
    coords_b = front_b.coords

    tree_b = KDTree(coords_b)
    dists_a2b, _ = tree_b.query(coords_a)
    mean_a2b = np.mean(dists_a2b)

    tree_a = KDTree(coords_a)
    dists_b2a, _ = tree_a.query(coords_b)
    mean_b2a = np.mean(dists_b2a)

    return 0.5 * (mean_a2b + mean_b2a)


def _match_fronts(
    fronts_a: FrontCollection,
    fronts_b: FrontCollection,
    max_dist: float,
) -> dict[str, str]:
    """Asignacion greedy de correspondencias entre dos conjuntos de frentes.

    Returns:
        Dict front_id_a -> front_id_b para las correspondencias encontradas.
    """
    if not fronts_a.fronts or not fronts_b.fronts:
        return {}

    # Calcular matriz de distancias
    pairs = []
    for fa in fronts_a:
        if fa.front_type == FrontType.INSTABILITY_LINE:
            continue
        for fb in fronts_b:
            if fb.front_type == FrontType.INSTABILITY_LINE:
                continue
            d = _mean_min_distance(fa, fb)
            if d < max_dist:
                pairs.append((d, fa.id, fb.id))

    # Greedy: asignar por distancia creciente
    pairs.sort(key=lambda x: x[0])
    used_a = set()
    used_b = set()
    result = {}

    for d, id_a, id_b in pairs:
        if id_a in used_a or id_b in used_b:
            continue
        result[id_a] = id_b
        used_a.add(id_a)
        used_b.add(id_b)

    return result


def _match_consecutive(
    all_fronts: list[FrontCollection],
    max_dist: float,
) -> list[dict[str, str]]:
    """Calcula correspondencias entre cada par consecutivo de timesteps.

    Returns:
        Lista de dicts, uno por cada par (t, t+1). matches[i] mapea
        ids de all_fronts[i] a ids de all_fronts[i+1].
    """
    matches = []
    for i in range(len(all_fronts) - 1):
        m = _match_fronts(all_fronts[i], all_fronts[i + 1], max_dist)
        logger.debug(
            "Matching timestep %d->%d: %d correspondencias", i, i + 1, len(m),
        )
        matches.append(m)
    return matches


# --- Persistencia ---


def _filter_by_persistence(
    all_fronts: list[FrontCollection],
    matches: list[dict[str, str]],
    center_idx: int,
    min_persistence: int,
) -> FrontCollection:
    """Filtra frentes del timestep central por persistencia temporal.

    Un frente del timestep central se conserva si tiene correspondencias
    en al menos min_persistence otros timesteps (hacia atras o hacia adelante).
    """
    center = all_fronts[center_idx]
    result = FrontCollection()
    result.valid_time = center.valid_time
    result.model_run = center.model_run

    for front in center:
        if front.front_type == FrontType.INSTABILITY_LINE:
            result.add(front)
            continue

        persistence_count = _count_persistence(
            front.id, all_fronts, matches, center_idx,
        )

        if persistence_count >= min_persistence:
            result.add(front)
        else:
            logger.debug(
                "Frente %s descartado (persistencia=%d < %d)",
                front.id, persistence_count, min_persistence,
            )

    return result


def _count_persistence(
    front_id: str,
    all_fronts: list[FrontCollection],
    matches: list[dict[str, str]],
    center_idx: int,
) -> int:
    """Cuenta en cuantos timesteps adicionales aparece un frente."""
    count = 0

    # Trazar hacia adelante: center -> center+1 -> ...
    current_id = front_id
    for i in range(center_idx, len(matches)):
        if current_id in matches[i]:
            next_id = matches[i][current_id]
            count += 1
            current_id = next_id
        else:
            break

    # Trazar hacia atras: center -> center-1 -> ...
    # matches[i] mapea ids de all_fronts[i] a all_fronts[i+1],
    # asi que necesitamos invertir para ir de i+1 a i
    current_id = front_id
    for i in range(center_idx - 1, -1, -1):
        # Invertir matches[i]: buscar que id en all_fronts[i] apunta a current_id
        inv = {v: k for k, v in matches[i].items()}
        if current_id in inv:
            prev_id = inv[current_id]
            count += 1
            current_id = prev_id
        else:
            break

    return count


# --- Clasificacion por movimiento ---


def _classify_persistent_fronts(
    persistent: FrontCollection,
    all_fronts: list[FrontCollection],
    matches: list[dict[str, str]],
    center_idx: int,
    ds_center: xr.Dataset,
    cfg: AppConfig,
):
    """Clasifica frentes persistentes por desplazamiento temporal.

    Para cada frente, busca su correspondencia en el timestep anterior
    y siguiente. El vector de desplazamiento (diferencia de centroides)
    se compara con la direccion del gradiente termico para distinguir
    frio (avanza hacia aire calido) de calido (avanza hacia aire frio).
    """
    ds_c = _remove_time_dim(ds_center)
    lat_name = "latitude" if "latitude" in ds_c.coords else "lat"
    lon_name = "longitude" if "longitude" in ds_c.coords else "lon"
    lats = ds_c[lat_name].values
    lons = ds_c[lon_name].values

    theta_w = compute_theta_w(ds_c)
    theta_w = smooth_field(theta_w, sigma=cfg.tfp.smooth_sigma)
    gx, gy = spherical_gradient(theta_w, lats, lons)

    classified = 0
    for front in persistent:
        if front.front_type == FrontType.INSTABILITY_LINE:
            continue

        # Buscar correspondencia anterior
        matched_prev = _find_matched_front(
            front.id, all_fronts, matches, center_idx, direction=-1,
        )
        # Buscar correspondencia posterior
        matched_next = _find_matched_front(
            front.id, all_fronts, matches, center_idx, direction=+1,
        )

        if matched_prev is None and matched_next is None:
            # Sin correspondencias para movimiento: fallback a adveccion
            continue

        ftype = _classify_by_movement(
            front, matched_prev, matched_next, lats, lons, gx, gy,
        )
        if ftype is not None:
            front.front_type = ftype
            classified += 1

    # Fallback: clasificar los restantes con adveccion instantanea
    unclassified = [
        f for f in persistent
        if f.front_type == FrontType.COLD
        and f.front_type != FrontType.INSTABILITY_LINE
    ]
    if unclassified:
        # Usar clasificador por adveccion para los no clasificados por movimiento
        persistent_copy = classify_fronts(persistent, ds_center, cfg)
        # Solo actualizar los que no fueron clasificados por movimiento
        for f in persistent_copy:
            orig = persistent.get_by_id(f.id)
            if orig is not None and orig.front_type == FrontType.COLD and classified == 0:
                orig.front_type = f.front_type

    logger.info("Clasificacion temporal: %d frentes clasificados por movimiento", classified)


def _find_matched_front(
    front_id: str,
    all_fronts: list[FrontCollection],
    matches: list[dict[str, str]],
    center_idx: int,
    direction: int,
) -> Front | None:
    """Busca la correspondencia de un frente en el timestep adyacente.

    Args:
        direction: -1 para timestep anterior, +1 para timestep siguiente.
    """
    if direction == +1:
        # matches[center_idx] mapea center -> center+1
        if center_idx < len(matches):
            next_id = matches[center_idx].get(front_id)
            if next_id is not None:
                return all_fronts[center_idx + 1].get_by_id(next_id)
    elif direction == -1:
        # matches[center_idx-1] mapea center-1 -> center
        if center_idx > 0:
            inv = {v: k for k, v in matches[center_idx - 1].items()}
            prev_id = inv.get(front_id)
            if prev_id is not None:
                return all_fronts[center_idx - 1].get_by_id(prev_id)
    return None


def _classify_by_movement(
    front: Front,
    matched_prev: Front | None,
    matched_next: Front | None,
    lats: np.ndarray,
    lons: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
) -> FrontType | None:
    """Clasifica un frente por su vector de desplazamiento vs gradiente termico.

    Calcula el centroide del frente y de sus correspondencias temporales.
    El vector de desplazamiento se compara con el gradiente de theta_w
    en el centroide del frente.

    Returns:
        FrontType clasificado, o None si la senal es ambigua.
    """
    # Centroide del frente actual
    c_lon = np.mean(front.lons)
    c_lat = np.mean(front.lats)

    # Vector de desplazamiento
    displacements = []
    if matched_prev is not None:
        prev_lon = np.mean(matched_prev.lons)
        prev_lat = np.mean(matched_prev.lats)
        # Desplazamiento prev -> center
        displacements.append((c_lon - prev_lon, c_lat - prev_lat))

    if matched_next is not None:
        next_lon = np.mean(matched_next.lons)
        next_lat = np.mean(matched_next.lats)
        # Desplazamiento center -> next
        displacements.append((next_lon - c_lon, next_lat - c_lat))

    if not displacements:
        return None

    # Vector de desplazamiento medio
    disp_lon = np.mean([d[0] for d in displacements])
    disp_lat = np.mean([d[1] for d in displacements])
    disp_norm = np.sqrt(disp_lon**2 + disp_lat**2)

    if disp_norm < 0.1:
        # Frente practicamente estacionario
        return FrontType.STATIONARY

    # Gradiente termico en el centroide
    j = np.argmin(np.abs(lats - c_lat))
    i = np.argmin(np.abs(lons - c_lon))
    j = np.clip(j, 0, gx.shape[0] - 1)
    i = np.clip(i, 0, gx.shape[1] - 1)

    grad_lon = gx[j, i]
    grad_lat = gy[j, i]
    grad_norm = np.sqrt(grad_lon**2 + grad_lat**2)

    if grad_norm < 1e-12:
        return None

    # Producto escalar normalizado: desplazamiento . gradiente
    # Si positivo: el frente se mueve en la direccion del gradiente
    # (hacia aire calido) -> frente frio
    # Si negativo: se mueve contra el gradiente -> frente calido
    dot = (disp_lon * grad_lon + disp_lat * grad_lat) / (disp_norm * grad_norm)

    # Evaluar consistencia a lo largo del frente (no solo centroide)
    cross_products = []
    for k in range(0, len(front.lats), max(1, len(front.lats) // 10)):
        lat_k, lon_k = front.lats[k], front.lons[k]
        jk = np.clip(np.argmin(np.abs(lats - lat_k)), 0, gx.shape[0] - 1)
        ik = np.clip(np.argmin(np.abs(lons - lon_k)), 0, gx.shape[1] - 1)
        gx_k, gy_k = gx[jk, ik], gy[jk, ik]
        gn_k = np.sqrt(gx_k**2 + gy_k**2)
        if gn_k > 1e-12:
            cross_products.append(
                (disp_lon * gx_k + disp_lat * gy_k) / (disp_norm * gn_k)
            )

    if not cross_products:
        return None

    mean_dot = np.mean(cross_products)

    # Umbrales de decision
    if mean_dot > 0.2:
        return FrontType.COLD
    elif mean_dot < -0.2:
        return FrontType.WARM
    elif abs(mean_dot) <= 0.2:
        # Senal mixta o debil: posiblemente ocluido
        # Verificar si hay mezcla de senales
        n_pos = sum(1 for c in cross_products if c > 0)
        n_neg = sum(1 for c in cross_products if c < 0)
        total = len(cross_products)
        if total > 0 and min(n_pos, n_neg) / total > 0.3:
            return FrontType.OCCLUDED

    return None
