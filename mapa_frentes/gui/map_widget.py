"""Widget de mapa con Cartopy embebido en PyQt5 via FigureCanvasQTAgg.

Soporta dos modos:
- Vista unica: un solo GeoAxes con cartografia detallada (50m)
- Vista mosaico: NxM GeoAxes con cartografia ligera (110m)
"""

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.ticker import MultipleLocator
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from mapa_frentes.config import AppConfig
from mapa_frentes.plotting.map_canvas import (
    create_map_figure, build_projection, apply_base_cartography,
)


class MapWidget(QWidget):
    """Widget PyQt que contiene un canvas Matplotlib con Cartopy."""

    def __init__(self, cfg: AppConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.mosaic_mode = False
        self.axes_dict = {}   # {field_name: GeoAxes} en modo mosaico
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig, self.ax = create_map_figure(self.cfg)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._background = None

    def switch_to_mosaic(self, field_names, nrows=2, ncols=3):
        """Cambia a vista mosaico con los campos dados."""
        projection = build_projection(self.cfg)
        self.fig.clear()

        axes_grid = self.fig.subplots(
            nrows, ncols,
            subplot_kw={"projection": projection},
        )
        if nrows == 1 and ncols == 1:
            all_axes = [axes_grid]
        else:
            all_axes = axes_grid.flatten()

        self.axes_dict = {}
        for i, ax in enumerate(all_axes):
            if i < len(field_names):
                name = field_names[i]
                self.axes_dict[name] = ax
                apply_base_cartography(ax, self.cfg, lightweight=True)
            else:
                ax.set_visible(False)

        self.fig.subplots_adjust(
            left=0.02, right=0.98, top=0.95, bottom=0.02,
            wspace=0.05, hspace=0.08,
        )

        self.ax = list(self.axes_dict.values())[0] if self.axes_dict else self.ax
        self.mosaic_mode = True
        self._background = None
        self.canvas.draw()

    def switch_to_single(self):
        """Vuelve a vista unica."""
        self.fig.clear()
        projection = build_projection(self.cfg)
        self.ax = self.fig.add_subplot(1, 1, 1, projection=projection)
        apply_base_cartography(self.ax, self.cfg, lightweight=False)

        gl = self.ax.gridlines(
            draw_labels=True, linewidth=0.3, color="gray",
            alpha=0.4, linestyle="--", x_inline=False, y_inline=False,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 7, "color": "gray"}
        gl.ylabel_style = {"size": 7, "color": "gray"}
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(10)

        self.axes_dict = {}
        self.mosaic_mode = False
        self._background = None
        self.canvas.draw()

    def get_valid_axes(self):
        """Devuelve lista de axes validos para edicion."""
        if self.mosaic_mode:
            return list(self.axes_dict.values())
        return [self.ax]

    def redraw(self):
        """Fuerza un redibujado completo del canvas."""
        self.canvas.draw()
        self._background = None

    def cache_background(self):
        """Cachea el fondo actual para blitting eficiente."""
        self.canvas.draw()
        if self.mosaic_mode:
            self._background = self.canvas.copy_from_bbox(self.fig.bbox)
        else:
            self._background = self.canvas.copy_from_bbox(self.ax.bbox)

    def restore_background(self):
        """Restaura el fondo cacheado."""
        if self._background is not None:
            self.canvas.restore_region(self._background)

    def blit(self):
        """Actualiza el area del canvas (blitting)."""
        if self.mosaic_mode:
            self.canvas.blit(self.fig.bbox)
        else:
            self.canvas.blit(self.ax.bbox)

    def clear_map(self):
        """Limpia el mapa y lo recrea desde cero."""
        self.fig.clear()
        self.fig, self.ax = create_map_figure(self.cfg)
        self.mosaic_mode = False
        self.axes_dict = {}
        self.redraw()
