"""Conectar puntos de frente en polilineas suaves.

Pipeline:
1. DBSCAN agrupa los puntos dispersos en clusters
2. Dentro de cada cluster, nearest-neighbor con preferencia angular
   (favorece continuar en la misma direccion, evita zigzag)
3. Corte por max_hop si hay saltos grandes
4. Suavizado con spline cubica para polilineas naturales
5. Douglas-Peucker simplifica ligeramente
6. Filtro de longitud minima (~700 km) para descartar fragmentos
"""

import logging

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString

logger = logging.getLogger(__name__)

# Longitud minima de un frente en grados (~300 km a 45N)
MIN_FRONT_LENGTH_DEG = 3.0

# Distancia maxima de salto entre puntos consecutivos (en grados)
MAX_HOP_DEG = 3.0

# Peso de la penalizacion angular (0=solo distancia, 1=puro angular)
ANGULAR_WEIGHT = 0.55


def cluster_and_connect(
    lats: np.ndarray,
    lons: np.ndarray,
    eps_deg: float = 1.5,
    min_samples: int = 4,
    min_points: int = 10,
    simplify_tol: float = 0.2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Agrupa puntos con DBSCAN y los conecta en polilineas suaves."""
    if len(lats) < min_points:
        return []

    coords = np.column_stack([lons, lats])
    db = DBSCAN(eps=eps_deg, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(coords)

    polylines = []
    unique_labels = set(labels)
    unique_labels.discard(-1)

    logger.info(
        "DBSCAN: %d clusters, %d puntos ruido de %d totales",
        len(unique_labels), np.sum(labels == -1), len(labels),
    )

    for label in sorted(unique_labels):
        mask = labels == label
        cluster_lats = lats[mask]
        cluster_lons = lons[mask]

        if len(cluster_lats) < min_points:
            continue

        segments = _directed_nearest_neighbor(
            cluster_lats, cluster_lons, max_hop=MAX_HOP_DEG
        )

        for seg_lats, seg_lons in segments:
            if len(seg_lats) < min_points:
                continue

            # Suavizar con spline cubica
            sm_lats, sm_lons = _smooth_spline(seg_lats, seg_lons)

            # Simplificar ligeramente
            sm_lats, sm_lons = _simplify_polyline(
                sm_lats, sm_lons, tolerance=simplify_tol
            )

            if len(sm_lats) < 3:
                continue

            # Filtro de longitud minima
            length = _polyline_length_deg(sm_lats, sm_lons)
            if length < MIN_FRONT_LENGTH_DEG:
                continue

            polylines.append((sm_lats, sm_lons))

    # Ordenar frentes de mas largo a mas corto
    polylines.sort(
        key=lambda p: _polyline_length_deg(p[0], p[1]), reverse=True
    )

    logger.info("Polilineas finales: %d", len(polylines))
    return polylines


def _directed_nearest_neighbor(
    lats: np.ndarray,
    lons: np.ndarray,
    max_hop: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Nearest-neighbor con preferencia de continuidad angular.

    Favorece vecinos que continuan en la misma direccion que el
    ultimo segmento, penalizando cambios de direccion bruscos.
    Esto evita los zigzags tipicos del NN puro.
    """
    n = len(lats)
    coords = np.column_stack([lons, lats])
    tree = KDTree(coords)

    visited = np.zeros(n, dtype=bool)
    segments = []

    while True:
        unvisited = np.where(~visited)[0]
        if len(unvisited) == 0:
            break

        # Semilla: punto mas occidental no visitado
        start = unvisited[np.argmin(lons[unvisited])]

        current_seg_idx = [start]
        visited[start] = True
        current = start
        prev_dir = None  # direccion del ultimo paso

        while True:
            dists, indices = tree.query(coords[current], k=min(30, n))

            best_idx = None
            best_score = np.inf

            for d, idx in zip(dists, indices):
                if idx == current or visited[idx]:
                    continue
                if d > max_hop or d < 1e-10:
                    continue

                # Score base: distancia
                score = d

                # Penalizacion angular: si tenemos direccion previa,
                # penalizar candidatos que requieren un giro brusco
                if prev_dir is not None:
                    new_dir = coords[idx] - coords[current]
                    new_dir_norm = np.linalg.norm(new_dir)
                    if new_dir_norm > 1e-10:
                        new_dir = new_dir / new_dir_norm
                        # cos(angulo) entre direccion previa y nueva
                        cos_angle = np.dot(prev_dir, new_dir)
                        # Penalizar giros: score *= (2 - cos) / 2
                        # cos=1 (recto) -> factor 0.5
                        # cos=0 (90 deg) -> factor 1.0
                        # cos=-1 (reversa) -> factor 1.5
                        angular_penalty = (2.0 - cos_angle) / 2.0
                        score = d * (
                            (1.0 - ANGULAR_WEIGHT)
                            + ANGULAR_WEIGHT * angular_penalty
                        )

                if score < best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            # Actualizar direccion
            new_dir = coords[best_idx] - coords[current]
            norm = np.linalg.norm(new_dir)
            if norm > 1e-10:
                prev_dir = new_dir / norm

            current_seg_idx.append(best_idx)
            visited[best_idx] = True
            current = best_idx

        if len(current_seg_idx) >= 5:
            idx = np.array(current_seg_idx)
            segments.append((lats[idx], lons[idx]))

    return segments


def _smooth_spline(
    lats: np.ndarray,
    lons: np.ndarray,
    num_points: int | None = None,
    smoothing: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Suaviza una polilinea con spline cubica parametrica.

    Produce una curva suave que pasa cerca de los puntos originales.
    El smoothing factor controla cuanto se aleja de los datos originales
    (valores mas altos = curvas mas suaves).
    """
    if len(lats) < 4:
        return lats, lons

    if num_points is None:
        # ~1 punto cada 0.3 grados para curvas mas finas
        total_len = _polyline_length_deg(lats, lons)
        num_points = max(int(total_len / 0.3), len(lats))
        num_points = min(num_points, 300)

    try:
        # Eliminar duplicados consecutivos
        points = np.column_stack([lons, lats])
        diffs = np.diff(points, axis=0)
        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        keep = np.concatenate([[True], seg_lengths > 1e-8])
        points = points[keep]
        lons_clean = points[:, 0]
        lats_clean = points[:, 1]

        if len(lats_clean) < 4:
            return lats, lons

        # Parametro acumulado (longitud de arco)
        diffs = np.diff(points, axis=0)
        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        u = np.zeros(len(points))
        u[1:] = np.cumsum(seg_lengths)
        u /= u[-1]

        # Ajustar spline
        k = min(3, len(lats_clean) - 1)
        s = smoothing * len(lats_clean)
        tck, _ = splprep([lons_clean, lats_clean], u=u, k=k, s=s)

        # Evaluar en puntos equiespaciados
        u_new = np.linspace(0, 1, num_points)
        smooth_lons, smooth_lats = splev(u_new, tck)

        return np.array(smooth_lats), np.array(smooth_lons)
    except Exception:
        return lats, lons


def _simplify_polyline(
    lats: np.ndarray,
    lons: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplifica una polilinea con Douglas-Peucker via Shapely."""
    if len(lats) < 2:
        return lats, lons

    line = LineString(zip(lons, lats))
    simplified = line.simplify(tolerance, preserve_topology=True)

    coords = np.array(simplified.coords)
    return coords[:, 1], coords[:, 0]


def _polyline_length_deg(lats: np.ndarray, lons: np.ndarray) -> float:
    """Calcula la longitud total de la polilinea en grados (aprox)."""
    dlat = np.diff(lats)
    dlon = np.diff(lons)
    return float(np.sum(np.sqrt(dlat**2 + dlon**2)))
