"""Widget de mapa con Cartopy embebido en PyQt5 via FigureCanvasQTAgg."""

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from mapa_frentes.config import AppConfig
from mapa_frentes.plotting.map_canvas import create_map_figure


class MapWidget(QWidget):
    """Widget PyQt que contiene un canvas Matplotlib con Cartopy."""

    def __init__(self, cfg: AppConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Crear figura y axes de Cartopy
        self.fig, self.ax = create_map_figure(self.cfg)

        # Crear canvas Qt
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)

        # Toolbar de navegacion Matplotlib (zoom, pan)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Cache para blitting
        self._background = None

    def redraw(self):
        """Fuerza un redibujado completo del canvas."""
        self.canvas.draw()
        self._background = None

    def cache_background(self):
        """Cachea el fondo actual para blitting eficiente."""
        self.canvas.draw()
        self._background = self.canvas.copy_from_bbox(self.ax.bbox)

    def restore_background(self):
        """Restaura el fondo cacheado."""
        if self._background is not None:
            self.canvas.restore_region(self._background)

    def blit(self):
        """Actualiza solo el area del axes (blitting)."""
        self.canvas.blit(self.ax.bbox)

    def clear_map(self):
        """Limpia el mapa y lo recrea desde cero."""
        self.fig.clear()
        self.fig, self.ax = create_map_figure(self.cfg)
        self.redraw()
