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


# Mapeo tipo de frente -> path effect de MetPy y color (estilo AEMET)
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
        "path_effect": StationaryFront,
        "color": "#006600",        # verde oscuro
        "label": "Frente estacionario",
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
    Los simbolos son mas grandes y espaciados que en la version por defecto
    para mejorar la legibilidad a la escala del mapa atlantico.
    """
    lw = cfg.plotting.front_linewidth
    artists = []

    for front in collection:
        if front.npoints < 2:
            continue

        style = FRONT_STYLES.get(front.front_type, FRONT_STYLES[FrontType.COLD])

        # Linea base (da cuerpo al frente, ligeramente mas gruesa)
        ax.plot(
            front.lons, front.lats,
            color=style["color"],
            linewidth=lw * 0.7,
            alpha=0.6,
            solid_capstyle="round",
            solid_joinstyle="round",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

        # Linea principal con path effects WMO
        # Simbolos mas grandes y espaciados para el dominio atlantico
        symbol_size = lw * 3.0
        symbol_spacing = lw * 8.0
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
            )],
        )
        line.front_id = front.id

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

        artists.append(line)

    return artists


def draw_front_legend(ax: Axes):
    """Dibuja la leyenda de tipos de frente en estilo AEMET."""
    handles = []
    for ftype, style in FRONT_STYLES.items():
        handle = Line2D(
            [0], [0],
            color=style["color"],
            linewidth=3,
            solid_capstyle="round",
            label=style["label"],
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
    )
