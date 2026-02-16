"""Conectar puntos de frente en polilineas suaves.

Pipeline:
1. DBSCAN agrupa los puntos dispersos en clusters
2. Dentro de cada cluster, nearest-neighbor con preferencia angular
   (favorece continuar en la misma direccion, evita zigzag)
3. Corte por max_hop si hay saltos grandes
4. Suavizado con spline cubica para polilineas naturales
5. Douglas-Peucker simplifica ligeramente
6. Fusion de segmentos cercanos y alineados
7. Filtro de longitud minima para descartar fragmentos
"""

import logging

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString

logger = logging.getLogger(__name__)


def cluster_and_connect(
    lats: np.ndarray,
    lons: np.ndarray,
    eps_deg: float = 1.8,
    min_samples: int = 3,
    min_points: int = 8,
    simplify_tol: float = 0.10,
    min_front_length_deg: float = 5.0,
    max_hop_deg: float = 3.0,
    angular_weight: float = 0.55,
    spline_smoothing: float = 0.6,
    merge_distance_deg: float = 3.0,
    max_fronts: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Agrupa puntos con DBSCAN y los conecta en polilineas suaves."""
    if len(lats) < min_points:
        return []

    coords = np.column_stack([lons, lats])
    db = DBSCAN(eps=eps_deg, min_samples=int(min_samples), metric="euclidean")
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
            cluster_lats, cluster_lons,
            max_hop=max_hop_deg,
            angular_weight=angular_weight,
        )

        for seg_lats, seg_lons in segments:
            if len(seg_lats) < min_points:
                continue

            # Suavizar con spline cubica
            sm_lats, sm_lons = _smooth_spline(
                seg_lats, seg_lons, smoothing=spline_smoothing
            )

            # Simplificar ligeramente
            sm_lats, sm_lons = _simplify_polyline(
                sm_lats, sm_lons, tolerance=simplify_tol
            )

            if len(sm_lats) < 3:
                continue

            polylines.append((sm_lats, sm_lons))

    # Fusionar segmentos cercanos y alineados
    if merge_distance_deg > 0:
        polylines = _merge_nearby_segments(
            polylines,
            merge_dist=merge_distance_deg,
            spline_smoothing=spline_smoothing,
        )

    # Filtro de longitud minima
    polylines = [
        p for p in polylines
        if _polyline_length_deg(p[0], p[1]) >= min_front_length_deg
    ]

    # Ordenar frentes de mas largo a mas corto
    polylines.sort(
        key=lambda p: _polyline_length_deg(p[0], p[1]), reverse=True
    )

    # Limitar numero maximo de frentes
    if max_fronts > 0 and len(polylines) > max_fronts:
        logger.info(
            "Truncando de %d a %d frentes (max_fronts)",
            len(polylines), max_fronts,
        )
        polylines = polylines[:max_fronts]

    logger.info("Polilineas finales: %d", len(polylines))
    return polylines


def _merge_nearby_segments(
    polylines: list[tuple[np.ndarray, np.ndarray]],
    merge_dist: float = 2.0,
    cos_threshold: float = 0.5,
    spline_smoothing: float = 0.6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fusiona segmentos cuyos extremos estan cerca y alineados.

    Dos segmentos se fusionan si:
    - La distancia entre un extremo de uno y un extremo del otro < merge_dist
    - Las direcciones en los extremos estan alineadas (cos > cos_threshold)
    Despues de fusionar, se re-suaviza con spline.
    """
    if len(polylines) < 2:
        return polylines

    merged = list(polylines)
    changed = True

    while changed:
        changed = False
        i = 0
        while i < len(merged):
            j = i + 1
            while j < len(merged):
                result = _try_merge(
                    merged[i], merged[j], merge_dist, cos_threshold
                )
                if result is not None:
                    # Re-suavizar el segmento fusionado
                    sm_lats, sm_lons = _smooth_spline(
                        result[0], result[1], smoothing=spline_smoothing
                    )
                    merged[i] = (sm_lats, sm_lons)
                    merged.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    return merged


def _try_merge(
    seg_a: tuple[np.ndarray, np.ndarray],
    seg_b: tuple[np.ndarray, np.ndarray],
    merge_dist: float,
    cos_threshold: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Intenta fusionar dos segmentos si sus extremos estan cerca y alineados.

    Devuelve el segmento fusionado o None si no se pueden fusionar.
    """
    lats_a, lons_a = seg_a
    lats_b, lons_b = seg_b

    if len(lats_a) < 2 or len(lats_b) < 2:
        return None

    # Extremos de A: inicio (0) y fin (-1)
    endpoints_a = [
        (lons_a[0], lats_a[0], lons_a[1] - lons_a[0], lats_a[1] - lats_a[0]),       # inicio, dir hacia adelante
        (lons_a[-1], lats_a[-1], lons_a[-1] - lons_a[-2], lats_a[-1] - lats_a[-2]),  # fin, dir desde anterior
    ]
    endpoints_b = [
        (lons_b[0], lats_b[0], lons_b[1] - lons_b[0], lats_b[1] - lats_b[0]),
        (lons_b[-1], lats_b[-1], lons_b[-1] - lons_b[-2], lats_b[-1] - lats_b[-2]),
    ]

    best_dist = merge_dist + 1
    best_combo = None

    for ia, ea in enumerate(endpoints_a):
        for ib, eb in enumerate(endpoints_b):
            dist = np.sqrt((ea[0] - eb[0])**2 + (ea[1] - eb[1])**2)
            if dist >= merge_dist:
                continue

            # Verificar alineacion angular
            dir_a = np.array([ea[2], ea[3]])
            dir_b = np.array([eb[2], eb[3]])
            norm_a = np.linalg.norm(dir_a)
            norm_b = np.linalg.norm(dir_b)
            if norm_a < 1e-10 or norm_b < 1e-10:
                continue

            dir_a = dir_a / norm_a
            dir_b = dir_b / norm_b
            cos_angle = abs(np.dot(dir_a, dir_b))

            if cos_angle >= cos_threshold and dist < best_dist:
                best_dist = dist
                best_combo = (ia, ib)

    if best_combo is None:
        return None

    ia, ib = best_combo
    # Concatenar en el orden correcto
    # ia=0: inicio de A conecta -> invertir A
    # ia=1: fin de A conecta -> A normal
    # ib=0: inicio de B conecta -> B normal
    # ib=1: fin de B conecta -> invertir B
    a_lats = lats_a[::-1] if ia == 0 else lats_a
    a_lons = lons_a[::-1] if ia == 0 else lons_a
    b_lats = lats_b if ib == 0 else lats_b[::-1]
    b_lons = lons_b if ib == 0 else lons_b[::-1]

    merged_lats = np.concatenate([a_lats, b_lats])
    merged_lons = np.concatenate([a_lons, b_lons])

    return (merged_lats, merged_lons)


def _directed_nearest_neighbor(
    lats: np.ndarray,
    lons: np.ndarray,
    max_hop: float,
    angular_weight: float = 0.55,
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
                            (1.0 - angular_weight)
                            + angular_weight * angular_penalty
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
    smoothing: float = 0.6,
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
