"""
Renderizado de frentes meteorologicos con simbologia WMO.

Usa MetPy path effects para dibujar la simbologia estandar
(triangulos para frente frio, semicirculos para calido, etc.).

Mejoras incluidas:
- Orientación estable del frente (desde el centro asociado hacia fuera) para que
  los símbolos no "cambien de lado" por invertir la polilínea.
- Spacing adaptativo para que siempre aparezcan símbolos también en frentes cortos.
"""

from __future__ import annotations

import numpy as np
import cartopy.crs as ccrs
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from metpy.plots import ColdFront, WarmFront, OccludedFront, StationaryFront

from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.models import FrontCollection, FrontType


# Mapeo tipo de frente -> path effect de MetPy y color (estilo AEMET/WMO)
FRONT_STYLES = {
    FrontType.COLD: {
        "path_effect": ColdFront,
        "color": "#0033BB",        # azul intenso (AEMET)
        "label": "Frente frio",
    },
    FrontType.WARM: {
        "path_effect": WarmFront,
        "color": "#CC0000",        # rojo (AEMET)
        "label": "Frente calido",
    },
    FrontType.OCCLUDED: {
        "path_effect": OccludedFront,
        "color": "#8800AA",        # purpura (AEMET)
        "label": "Frente ocluido",
    },
    FrontType.COLD_OCCLUDED: {
        "path_effect": OccludedFront,
        "color": "#6600CC",        # purpura-azulado
        "label": "Oclusión fría",
    },
    FrontType.WARM_OCCLUDED: {
        "path_effect": OccludedFront,
        "color": "#AA0088",        # purpura-rojizo
        "label": "Oclusión cálida",
    },
    FrontType.WARM_SECLUSION: {
        "path_effect": OccludedFront,
        "color": "#DD0066",        # magenta
        "label": "Seclusión cálida",
    },
    FrontType.STATIONARY: {
        "path_effect": None,       # handled specially with WMO bicolor
        "color": "#006600",        # fallback (unused for line)
        "label": "Frente estacionario",
    },
    FrontType.INSTABILITY_LINE: {
        "path_effect": None,       # dash-dot line, no MetPy symbols
        "color": "#CC4400",        # naranja oscuro
        "label": "Linea de inestabilidad",
    },
}


def _orient_front_outward(front) -> None:
    """
    Asegura que el frente esté orientado desde el centro (extremo asociado) hacia fuera.

    MetPy decide el "lado" de los símbolos en función del sentido de la polilínea.
    Si el frente está guardado al revés, los triángulos/semicírculos salen al lado contrario.

    Regla:
    - si association_end == "end": invertimos para que el extremo asociado quede como "start"
    """
    if getattr(front, "association_end", None) == "end":
        front.lons = front.lons[::-1]
        front.lats = front.lats[::-1]
        if hasattr(front, "scores") and isinstance(front.scores, (list, np.ndarray)):
            front.scores = front.scores[::-1]


def _adaptive_spacing(
    ax: Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    base_spacing: float,
    min_symbols: int = 1,
) -> float:
    """
    Calcula un spacing adaptativo para los símbolos WMO.

    Problema: MetPy coloca símbolos cada `spacing`. Si el frente es corto en pantalla,
    puede no caber ningún símbolo. Aquí reducimos spacing en frentes cortos para que
    aparezca al menos `min_symbols` (o, en la práctica, al menos 1).

    Nota: no intentamos convertir exactamente a puntos/píxeles; buscamos una regla robusta
    y consistente visualmente.
    """
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    if lons.size < 2 or lats.size < 2:
        return float(base_spacing)

    pts = ax.projection.transform_points(ccrs.PlateCarree(), lons, lats)
    x = pts[:, 0]
    y = pts[:, 1]
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return float(base_spacing)

    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    length = float(np.nansum(seg))
    if length <= 0:
        return float(base_spacing)

    # Heurística:
    # - Para frentes largos: spacing ~ base_spacing
    # - Para frentes cortos: spacing baja, con un mínimo razonable
    # El factor 0.25 controla cuán rápido baja spacing en frentes cortos.
    target = 0.25 * length / (min_symbols + 1)

    # Limita para no poner símbolos absurdamente densos o demasiado separados.
    spacing = max(4.0, min(float(base_spacing), float(target)))
    return float(spacing)


def draw_fronts(
    ax: Axes,
    collection: FrontCollection,
    cfg: AppConfig,
    highlight_id: str | None = None,
    filter_fronts: list | None = None,
    show_importance: bool = False,
    skip_orient: bool = False,
):
    """Dibuja todos los frentes de la coleccion en el mapa.

    Estilo AEMET: linea base con color y path effects WMO encima.

    Args:
        ax: Axes de matplotlib.
        collection: Colección de frentes.
        cfg: Configuración.
        highlight_id: ID del frente a resaltar (opcional).
        filter_fronts: Lista de frentes a dibujar (None = todos).
        show_importance: Si True, aplica estilo diferenciado a primarios/secundarios.
        skip_orient: Si True, no aplica orientación automática (para export web
                     donde el usuario ya ajustó la dirección con flip).
    """
    lw = cfg.plotting.front_linewidth
    symbol_size = cfg.plotting.front_symbol_size
    base_spacing = cfg.plotting.front_symbol_spacing

    artists = []

    fronts_to_draw = filter_fronts if filter_fronts is not None else collection.fronts

    for front in fronts_to_draw:
        if front.npoints < 2:
            continue

        # (1) Orientación estable para evitar que los símbolos "salten" de lado
        if not skip_orient:
            _orient_front_outward(front)

        style = FRONT_STYLES.get(front.front_type, FRONT_STYLES[FrontType.COLD])

        # (2) Estilo diferenciado para frentes secundarios
        alpha = 1.0
        linewidth_multiplier = 1.0
        if show_importance and hasattr(front, "is_primary"):
            if not front.is_primary:
                alpha = 0.4
                linewidth_multiplier = 0.6

        # (3) Spacing adaptativo: asegura símbolos también en frentes cortos
        spacing = _adaptive_spacing(
            ax=ax,
            lons=np.asarray(front.lons),
            lats=np.asarray(front.lats),
            base_spacing=base_spacing,
            min_symbols=1,
        )

        # Resaltar frente seleccionado (para la GUI)
        if highlight_id and front.id == highlight_id:
            ax.plot(
                front.lons,
                front.lats,
                color="yellow",
                linewidth=lw + 3,
                alpha=0.5,
                solid_capstyle="round",
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

        # --- Línea de inestabilidad: estilo dash-dot, sin path_effect ---
        if front.front_type == FrontType.INSTABILITY_LINE:
            line, = ax.plot(
                front.lons,
                front.lats,
                color=style["color"],
                linewidth=lw * linewidth_multiplier,
                linestyle=(0, (5, 3, 1, 3)),  # dash-dot
                solid_capstyle="round",
                alpha=alpha,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            line.front_id = front.id
            artists.append(line)
            continue

        # --- Frente estacionario: bicolor WMO (rojo + azul) ---
        if front.front_type == FrontType.STATIONARY:
            # Línea base
            ax.plot(
                front.lons,
                front.lats,
                color="#CC0000",
                linewidth=lw * 0.5 * linewidth_multiplier,
                alpha=0.5 * alpha,
                solid_capstyle="round",
                solid_joinstyle="round",
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

            # Símbolos WMO bicolor: semicirculos rojos + triángulos azules
            flip = getattr(front, "flip_symbols", False)

            line, = ax.plot(
                front.lons,
                front.lats,
                color="#CC0000",
                linewidth=lw * linewidth_multiplier,
                solid_capstyle="round",
                solid_joinstyle="round",
                alpha=alpha,
                transform=ccrs.PlateCarree(),
                zorder=5,
                path_effects=[
                    StationaryFront(
                        size=symbol_size,
                        spacing=spacing,
                        colors=("#CC0000", "#0033BB"),
                        flip=flip,
                    )
                ],
            )
            line.front_id = front.id
            artists.append(line)
            continue

        # --- Frentes normales (frío, cálido, ocluido) ---
        # Línea base (da cuerpo al frente)
        ax.plot(
            front.lons,
            front.lats,
            color=style["color"],
            linewidth=lw * 0.5 * linewidth_multiplier,
            alpha=0.5 * alpha,
            solid_capstyle="round",
            solid_joinstyle="round",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

        flip = getattr(front, "flip_symbols", False)

        # Frentes fríos: MetPy pone triángulos al revés por defecto con orientación centro→fuera
        if front.front_type == FrontType.COLD:
            flip = not flip

        line, = ax.plot(
            front.lons,
            front.lats,
            color=style["color"],
            linewidth=lw * linewidth_multiplier,
            solid_capstyle="round",
            solid_joinstyle="round",
            alpha=alpha,
            transform=ccrs.PlateCarree(),
            zorder=5,
            path_effects=[
                style["path_effect"](
                    size=symbol_size,
                    spacing=spacing,
                    flip=flip,
                )
            ],
        )
        line.front_id = front.id
        artists.append(line)

    return artists


def draw_front_legend(ax: Axes, cfg: AppConfig):
    """Leyenda horizontal debajo del mapa (compacta, estilo operativo)."""
    fig = ax.figure

    handles = [
        Line2D([0], [0], color=FRONT_STYLES[FrontType.COLD]["color"], lw=3, label="Frente frío"),
        Line2D([0], [0], color=FRONT_STYLES[FrontType.WARM]["color"], lw=3, label="Frente cálido"),
        Line2D([0], [0], color=FRONT_STYLES[FrontType.OCCLUDED]["color"], lw=3, label="Frente ocluido"),
        Line2D([0], [0], color="#7B68EE", lw=3, label="Frente estacionario"),
        Line2D(
            [0],
            [0],
            color=FRONT_STYLES[FrontType.INSTABILITY_LINE]["color"],
            lw=3,
            linestyle=(0, (5, 3, 1, 3)),
            label="Línea de inestabilidad",
        ),
    ]

    # Eliminar leyenda previa si existe
    if hasattr(fig, "_front_legend") and fig._front_legend:
        try:
            fig._front_legend.remove()
        except Exception:
            pass

    fig._front_legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=len(handles),
        frameon=True,
        fancybox=True,
        fontsize=6.8,
        columnspacing=0.9,
        handlelength=1.9,
        handletextpad=0.6,
        borderpad=0.25,
        labelspacing=0.35,
        framealpha=0.9,
    )


def draw_association_lines(
    ax: Axes,
    collection: FrontCollection,
    centers: list,
    cfg: AppConfig,
):
    """Dibuja lineas punteadas de asociacion frente-borrasca.

    Para cada frente con center_id, dibuja una linea punteada gris fina
    desde el extremo indicado por association_end hasta el centro L.
    Solo si la distancia > 0.3 deg (si ya esta conectado, no dibujar).
    """
    center_map = {c.id: c for c in centers}

    for front in collection:
        if not front.center_id or front.center_id not in center_map:
            continue

        center = center_map[front.center_id]

        # Determinar extremo del frente
        if front.association_end == "start":
            f_lon, f_lat = front.lons[0], front.lats[0]
        elif front.association_end == "end":
            f_lon, f_lat = front.lons[-1], front.lats[-1]
        else:
            # Sin association_end, usar extremo mas cercano
            d_start = np.sqrt((front.lats[0] - center.lat) ** 2 + (front.lons[0] - center.lon) ** 2)
            d_end = np.sqrt((front.lats[-1] - center.lat) ** 2 + (front.lons[-1] - center.lon) ** 2)
            if d_start < d_end:
                f_lon, f_lat = front.lons[0], front.lats[0]
            else:
                f_lon, f_lat = front.lons[-1], front.lats[-1]

        # Solo dibujar si hay distancia significativa
        dist = np.sqrt((f_lat - center.lat) ** 2 + (f_lon - center.lon) ** 2)
        if dist < 0.3:
            continue

        ax.plot(
            [f_lon, center.lon],
            [f_lat, center.lat],
            color="gray",
            linewidth=0.5,
            linestyle=":",
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
