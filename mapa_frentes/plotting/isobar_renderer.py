"""Renderizado de isobaras y etiquetas A/B en el mapa."""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.axes import Axes

from mapa_frentes.config import AppConfig
from mapa_frentes.analysis.pressure_centers import PressureCenter


def draw_isobars(
    ax: Axes,
    msl_smooth: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    levels: np.ndarray,
    cfg: AppConfig,
):
    """Dibuja las isobaras sobre el mapa.

    Isobaras normales en gris fino, isobaras cada 20 hPa mas gruesas.
    Etiquetas en todas las isobaras normales.
    """
    iso_cfg = cfg.isobars
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Isobaras normales
    cs = ax.contour(
        lon2d, lat2d, msl_smooth,
        levels=levels,
        colors=iso_cfg.color,
        linewidths=iso_cfg.linewidth,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.clabel(
        cs,
        inline=True,
        fontsize=iso_cfg.label_fontsize,
        fmt="%d",
    )

    # Isobaras maestras (cada 20 hPa) mas gruesas
    master_levels = levels[levels % 20 == 0]
    if len(master_levels) > 0:
        ax.contour(
            lon2d, lat2d, msl_smooth,
            levels=master_levels,
            colors=iso_cfg.color,
            linewidths=iso_cfg.linewidth * 2.0,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )


def draw_pressure_labels(
    ax: Axes,
    centers: list[PressureCenter],
    cfg: AppConfig,
):
    """Dibuja etiquetas de centros de presion estilo AEMET.

    - Primarios: B/A mayuscula, tamano grande
    - Secundarios: b/a minuscula, tamano menor
    - Si tiene nombre (borrasca nombrada): se muestra junto a la etiqueta
    Solo se dibujan centros dentro del area de visualizacion.
    """
    pc_cfg = cfg.pressure_centers
    area = cfg.area  # Area de visualizacion (sin padding)

    for center in centers:
        # Filtrar centros fuera del area visible
        if (center.lon < area.lon_min or center.lon > area.lon_max
                or center.lat < area.lat_min or center.lat > area.lat_max):
            continue

        color = pc_cfg.h_color if center.type == "H" else pc_cfg.l_color
        base_label = pc_cfg.high_label if center.type == "H" else pc_cfg.low_label

        if center.primary:
            label = base_label.upper()
            fontsize = pc_cfg.fontsize + 2
        else:
            label = base_label.lower()
            fontsize = pc_cfg.fontsize - 2

        # Letra A/B (o a/b) con halo blanco
        ax.text(
            center.lon, center.lat,
            label,
            transform=ccrs.PlateCarree(),
            fontsize=fontsize,
            fontweight=pc_cfg.fontweight,
            color=color,
            ha="center",
            va="center",
            zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
            ),
        )

        # Nombre de la borrasca (si tiene)
        if center.name:
            ax.text(
                center.lon + 1.0, center.lat,
                center.name,
                transform=ccrs.PlateCarree(),
                fontsize=pc_cfg.fontsize - 2,
                fontweight="bold",
                fontstyle="italic",
                color="black",
                ha="left",
                va="center",
                zorder=7,
            )

        # Valor de presion debajo
        value_fontsize = pc_cfg.fontsize - 3 if center.primary else pc_cfg.fontsize - 5
        ax.text(
            center.lon, center.lat - 1.4,
            f"{center.value:.0f}",
            transform=ccrs.PlateCarree(),
            fontsize=value_fontsize,
            fontweight="bold",
            color=color,
            ha="center",
            va="top",
            zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
            ),
        )
