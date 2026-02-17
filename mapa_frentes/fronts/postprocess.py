"""
Post-procesado de frentes: unión de segmentos (linking), filtrado y simplificación.

Objetivo:
- Reducir fragmentación típica del TFP (segmentos rotos).
- Quitar micro-frentes.
- Suavizar/simplificar geometría para un look "operativo".
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple, Dict

import numpy as np

from mapa_frentes.fronts.models import Front, FrontCollection, FrontType


# -------------------------
# Utilidades geo
# -------------------------

EARTH_RADIUS_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float(EARTH_RADIUS_KM * c)

def polyline_length_km(lats: np.ndarray, lons: np.ndarray) -> float:
    if len(lats) < 2:
        return 0.0
    total = 0.0
    for i in range(len(lats) - 1):
        total += haversine_km(lats[i], lons[i], lats[i+1], lons[i+1])
    return float(total)

def endpoint_vectors(front: Front) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve vector tangente (unitario) en el inicio y final del frente.
    """
    coords = front.coords
    if coords.shape[0] < 2:
        v0 = np.array([1.0, 0.0])
        v1 = np.array([1.0, 0.0])
        return v0, v1
    # inicio: punto1 - punto0
    a0 = coords[1] - coords[0]
    # final: ultimo - penultimo
    a1 = coords[-1] - coords[-2]

    def _unit(v):
        n = np.linalg.norm(v)
        if n == 0:
            return np.array([1.0, 0.0])
        return v / n

    return _unit(a0), _unit(a1)

def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    # ángulo entre vectores, en grados
    uu = u / (np.linalg.norm(u) + 1e-12)
    vv = v / (np.linalg.norm(v) + 1e-12)
    dot = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


# -------------------------
# Simplificación Douglas-Peucker (en grados)
# -------------------------

def _perp_dist(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    # distancia perpendicular de 'point' a segmento AB (en el plano lon/lat)
    ab = b - a
    if np.allclose(ab, 0):
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / (np.dot(ab, ab) + 1e-12))
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))

def douglas_peucker(coords: np.ndarray, tol: float) -> np.ndarray:
    """
    Simplifica una polilínea (coords Nx2 [lon,lat]) con tolerancia 'tol' en grados.
    """
    if coords.shape[0] <= 2:
        return coords

    a = coords[0]
    b = coords[-1]
    dmax = -1.0
    idx = -1
    for i in range(1, coords.shape[0] - 1):
        d = _perp_dist(coords[i], a, b)
        if d > dmax:
            dmax = d
            idx = i

    if dmax > tol:
        left = douglas_peucker(coords[: idx + 1], tol)
        right = douglas_peucker(coords[idx:], tol)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([a, b])


# -------------------------
# Linking de segmentos
# -------------------------

def _endpoints(front: Front) -> Dict[str, Tuple[float, float]]:
    # devuelve (lat,lon) en start/end
    return {
        "start": (float(front.lats[0]), float(front.lons[0])),
        "end": (float(front.lats[-1]), float(front.lons[-1])),
    }

def _reverse_front(front: Front) -> Front:
    # devuelve una copia con puntos invertidos
    return replace(front, lats=front.lats[::-1].copy(), lons=front.lons[::-1].copy())

def _concat_fronts(a: Front, b: Front) -> Front:
    """
    Concatena a + b (asumiendo que el final de a conecta con el inicio de b).
    Mantiene tipo/metadata de 'a' por defecto.
    """
    lats = np.concatenate([a.lats, b.lats])
    lons = np.concatenate([a.lons, b.lons])

    out = replace(a, lats=lats, lons=lons)
    # Nota: podrías combinar scores aquí si los usas
    return out

def link_front_segments(
    collection: FrontCollection,
    max_gap_km: float = 200.0,
    max_angle_deg: float = 30.0,
    require_same_type: bool = True,
) -> FrontCollection:
    """
    Une frentes fragmentados si sus extremos están cerca y la orientación es compatible.

    - max_gap_km: distancia máxima entre extremos a unir
    - max_angle_deg: ángulo máximo entre tangentes para permitir unión
    - require_same_type: si True, solo une frentes del mismo tipo
    """
    fronts = list(collection.fronts)
    used = np.zeros(len(fronts), dtype=bool)

    # Precalcular endpoints y tangentes
    ends = [_endpoints(f) for f in fronts]
    tang0 = []  # vector en start
    tang1 = []  # vector en end
    for f in fronts:
        v_start, v_end = endpoint_vectors(f)
        tang0.append(v_start)
        tang1.append(v_end)

    out_fronts: List[Front] = []

    for i in range(len(fronts)):
        if used[i]:
            continue

        current = fronts[i]
        used[i] = True

        # iremos intentando extender por ambos lados
        extended = True
        while extended:
            extended = False

            # extremos actuales
            cur_start = (float(current.lats[0]), float(current.lons[0]))
            cur_end   = (float(current.lats[-1]), float(current.lons[-1]))

            best_j = None
            best_mode = None
            best_dist = 1e9

            # buscar el mejor candidato para unir a un extremo (start o end)
            for j in range(len(fronts)):
                if used[j] or j == i:
                    continue
                cand = fronts[j]
                if require_same_type and cand.front_type != current.front_type:
                    continue

                # evaluar 4 combinaciones:
                # current_end -> cand_start (normal)
                d1 = haversine_km(cur_end[0], cur_end[1], ends[j]["start"][0], ends[j]["start"][1])
                # current_end -> cand_end (cand invertido)
                d2 = haversine_km(cur_end[0], cur_end[1], ends[j]["end"][0], ends[j]["end"][1])
                # cand_end -> current_start (unir por el otro lado)
                d3 = haversine_km(ends[j]["end"][0], ends[j]["end"][1], cur_start[0], cur_start[1])
                # cand_start -> current_start (cand invertido)
                d4 = haversine_km(ends[j]["start"][0], ends[j]["start"][1], cur_start[0], cur_start[1])

                # Para cada caso, comprobar ángulo usando tangentes:
                # Caso 1: current(end) con cand(start): comparar tang1(current) con tang0(cand)
                if d1 < best_dist and d1 <= max_gap_km:
                    ang = angle_deg(tang1_for(current), tang0[j])
                    if ang <= max_angle_deg:
                        best_dist, best_j, best_mode = d1, j, ("append", "normal")

                # Caso 2: current(end) con cand(end) (invertir cand): comparar tang1(current) con (-tang1(cand)) ~ tang0(cand_rev)
                if d2 < best_dist and d2 <= max_gap_km:
                    ang = angle_deg(tang1_for(current), -tang1[j])
                    if ang <= max_angle_deg:
                        best_dist, best_j, best_mode = d2, j, ("append", "reverse")

                # Caso 3: cand(end) con current(start): prepend cand normal: comparar (-tang1(cand)) con tang0(current)
                if d3 < best_dist and d3 <= max_gap_km:
                    ang = angle_deg(-tang1[j], tang0_for(current))
                    if ang <= max_angle_deg:
                        best_dist, best_j, best_mode = d3, j, ("prepend", "normal")

                # Caso 4: cand(start) con current(start) (invertir cand y prepend): comparar tang0(cand) con tang0(current) (pero cand invertido -> -tang0)
                if d4 < best_dist and d4 <= max_gap_km:
                    ang = angle_deg(tang0[j], -tang0_for(current))
                    if ang <= max_angle_deg:
                        best_dist, best_j, best_mode = d4, j, ("prepend", "reverse")

            if best_j is not None:
                cand = fronts[best_j]
                used[best_j] = True

                cand_use = cand if best_mode[1] == "normal" else _reverse_front(cand)

                if best_mode[0] == "append":
                    current = _concat_fronts(current, cand_use)
                else:
                    current = _concat_fronts(cand_use, current)

                extended = True  # seguir intentando unir

        out_fronts.append(current)

    new_col = FrontCollection(
        fronts=out_fronts,
        valid_time=collection.valid_time,
        model_run=collection.model_run,
        description=collection.description,
        metadata=dict(collection.metadata) if collection.metadata else {},
    )
    return new_col


def tang0_for(front: Front) -> np.ndarray:
    v0, _ = endpoint_vectors(front)
    return v0

def tang1_for(front: Front) -> np.ndarray:
    _, v1 = endpoint_vectors(front)
    return v1


# -------------------------
# Filtrado y simplificación
# -------------------------

def filter_and_simplify(
    collection: FrontCollection,
    min_length_km: float = 250.0,
    simplify_tol_deg: float = 0.12,
) -> FrontCollection:
    """
    - Descarta frentes cortos.
    - Simplifica geometría.
    """
    out: List[Front] = []
    for f in collection.fronts:
        L = polyline_length_km(f.lats, f.lons)
        if L < min_length_km:
            continue

        coords = f.coords
        coords2 = douglas_peucker(coords, tol=simplify_tol_deg)

        # Evitar que quede en 2 puntos demasiado cerca
        if coords2.shape[0] >= 2:
            f2 = replace(f)
            f2.set_coords(coords2)
            out.append(f2)

    return FrontCollection(
        fronts=out,
        valid_time=collection.valid_time,
        model_run=collection.model_run,
        description=collection.description,
        metadata=dict(collection.metadata) if collection.metadata else {},
    )
