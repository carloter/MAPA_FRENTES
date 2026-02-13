"""Renderizado de frentes meteorologicos con simbologia WMO.

Usa MetPy path effects para dibujar la simbologia estandar
(triangulos para frente frio, semicirculos para calido, etc.).
"""

import cartopy.crs as ccrs
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from metpy.plots import ColdFront, WarmFront, OccludedFront, StationaryFront

from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType


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


def draw_fronts(
    ax: Axes,
    collection: FrontCollection,
    cfg: AppConfig,
    highlight_id: str | None = None,
):
    """Dibuja todos los frentes de la coleccion en el mapa.

    Estilo AEMET: linea base con color y path effects WMO encima.
    Proporciones ajustadas para buena legibilidad en mapas atlanticos.
    """
    lw = cfg.plotting.front_linewidth
    symbol_size = cfg.plotting.front_symbol_size
    symbol_spacing = cfg.plotting.front_symbol_spacing
    artists = []

    for front in collection:
        if front.npoints < 2:
            continue

        style = FRONT_STYLES.get(front.front_type, FRONT_STYLES[FrontType.COLD])

        # Resaltar frente seleccionado (para la GUI)
        if highlight_id and front.id == highlight_id:
            ax.plot(
                front.lons, front.lats,
                color="yellow",
                linewidth=lw + 3,
                alpha=0.5,
                solid_capstyle="round",
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

        # --- Linea de inestabilidad: estilo dash-dot, sin path_effect ---
        if front.front_type == FrontType.INSTABILITY_LINE:
            line, = ax.plot(
                front.lons, front.lats,
                color=style["color"],
                linewidth=lw,
                linestyle=(0, (5, 3, 1, 3)),  # dash-dot
                solid_capstyle="round",
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            line.front_id = front.id
            artists.append(line)
            continue

        # --- Frente estacionario: bicolor WMO (rojo + azul) ---
        if front.front_type == FrontType.STATIONARY:
            # Linea base
            ax.plot(
                front.lons, front.lats,
                color="#CC0000",
                linewidth=lw * 0.5,
                alpha=0.5,
                solid_capstyle="round",
                solid_joinstyle="round",
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
            # Simbolos WMO bicolor: semicirculos rojos + triangulos azules
            flip = getattr(front, "flip_symbols", False)
            line, = ax.plot(
                front.lons, front.lats,
                color="#CC0000",
                linewidth=lw,
                solid_capstyle="round",
                solid_joinstyle="round",
                transform=ccrs.PlateCarree(),
                zorder=5,
                path_effects=[StationaryFront(
                    size=symbol_size,
                    spacing=symbol_spacing,
                    colors=("#CC0000", "#0033BB"),
                    flip=flip,
                )],
            )
            line.front_id = front.id
            artists.append(line)
            continue

        # --- Frentes normales (frio, calido, ocluido) ---
        # Linea base (da cuerpo al frente)
        ax.plot(
            front.lons, front.lats,
            color=style["color"],
            linewidth=lw * 0.5,
            alpha=0.5,
            solid_capstyle="round",
            solid_joinstyle="round",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

        # Linea principal con path effects WMO
        flip = getattr(front, "flip_symbols", False)
        line, = ax.plot(
            front.lons, front.lats,
            color=style["color"],
            linewidth=lw,
            solid_capstyle="round",
            solid_joinstyle="round",
            transform=ccrs.PlateCarree(),
            zorder=5,
            path_effects=[style["path_effect"](
                size=symbol_size,
                spacing=symbol_spacing,
                flip=flip,
            )],
        )
        line.front_id = front.id
        artists.append(line)

    return artists


def draw_front_legend(ax: Axes, cfg: AppConfig):
    """Dibuja la leyenda de tipos de frente con simbolos WMO reales."""
    lw = cfg.plotting.front_linewidth
    symbol_size = cfg.plotting.front_symbol_size
    symbol_spacing = cfg.plotting.front_symbol_spacing
    handles = []

    for ftype, style in FRONT_STYLES.items():
        if ftype == FrontType.INSTABILITY_LINE:
            # Linea dash-dot sin path_effect
            handle = Line2D(
                [0], [0],
                color=style["color"],
                linewidth=lw,
                linestyle=(0, (5, 3, 1, 3)),
                solid_capstyle="round",
                label=style["label"],
            )
        elif ftype == FrontType.STATIONARY:
            handle = Line2D(
                [0], [0],
                color="#CC0000",
                linewidth=lw,
                solid_capstyle="round",
                label=style["label"],
                path_effects=[StationaryFront(
                    size=symbol_size,
                    spacing=symbol_spacing,
                    colors=("#CC0000", "#0033BB"),
                )],
            )
        else:
            handle = Line2D(
                [0], [0],
                color=style["color"],
                linewidth=lw,
                solid_capstyle="round",
                label=style["label"],
                path_effects=[style["path_effect"](
                    size=symbol_size,
                    spacing=symbol_spacing,
                )],
            )
        handles.append(handle)

    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#AAAAAA",
        fancybox=True,
        shadow=True,
        handlelength=4.0,
    )
