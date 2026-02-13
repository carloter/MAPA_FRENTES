"""Edicion interactiva de frentes meteorologicos.

Modos de edicion:
- navigate: Navegacion normal (zoom, pan via toolbar Matplotlib)
- select: Click en un frente para seleccionarlo
- drag: Mover vertices del frente seleccionado
- add: Click para colocar puntos, doble-click para finalizar
- delete: Click en un frente para eliminarlo

Usa blitting de Matplotlib para rendimiento al arrastrar.
"""

import logging

import numpy as np
import cartopy.crs as ccrs
from matplotlib.lines import Line2D

from mapa_frentes.fronts.models import Front, FrontType

logger = logging.getLogger(__name__)

# Distancia maxima en pixels para considerar un click "cerca" de un frente
PICK_TOLERANCE_PX = 15


class FrontEditor:
    """Editor interactivo de frentes sobre el MapWidget.

    Se conecta a los eventos de Matplotlib (press, release, motion)
    y modifica los frentes de MainWindow.
    """

    def __init__(self, main_window):
        self.mw = main_window
        self.canvas = main_window.map_widget.canvas
        self.ax = main_window.map_widget.ax

        # Estado
        self.mode = "navigate"
        self.selected_front_id = None
        self.current_front_type = FrontType.COLD
        self._dragging = False
        self._drag_vertex_idx = None
        self._drag_front = None

        # Estado para modo "add"
        self._adding_points_lons = []
        self._adding_points_lats = []
        self._add_line = None
        self._add_markers = None

        # Artistas temporales para feedback visual
        self._vertex_markers = None
        self._highlight_line = None

        # Conectar eventos
        self._cids = []
        self._connect_events()

    def _connect_events(self):
        self._cids.append(
            self.canvas.mpl_connect("button_press_event", self._on_press)
        )
        self._cids.append(
            self.canvas.mpl_connect("button_release_event", self._on_release)
        )
        self._cids.append(
            self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        )

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids.clear()

    def set_mode(self, mode: str):
        """Cambia el modo de edicion."""
        # Finalizar modo anterior si es necesario
        if self.mode == "add" and mode != "add":
            self._finalize_add()
        self.mode = mode
        self._dragging = False

        if mode in ("navigate",):
            self._clear_temp_artists()

        logger.debug("Modo de edicion: %s", mode)

    def _event_to_lonlat(self, event):
        """Convierte coordenadas del evento (en la proyeccion del axes) a lon/lat."""
        return ccrs.PlateCarree().transform_point(
            event.xdata, event.ydata, self.ax.projection
        )

    # --- Event handlers ---

    def _on_press(self, event):
        if event.inaxes != self.ax or self.mode == "navigate":
            return
        if event.button != 1:  # solo boton izquierdo
            return

        lon, lat = self._event_to_lonlat(event)

        if self.mode == "select":
            self._handle_select(event, lon, lat)
        elif self.mode == "drag":
            self._handle_drag_start(event, lon, lat)
        elif self.mode == "add":
            self._handle_add_point(event, lon, lat)
        elif self.mode == "delete":
            self._handle_delete(event, lon, lat)

    def _on_release(self, event):
        if self._dragging:
            self._dragging = False
            self._drag_vertex_idx = None
            # Redibujar completamente despues del arrastre
            self.mw._refresh_map()

    def _on_motion(self, event):
        if not self._dragging or event.inaxes != self.ax:
            return
        if self._drag_front is None or self._drag_vertex_idx is None:
            return

        lon, lat = self._event_to_lonlat(event)

        # Actualizar coordenada del vertice arrastrado
        self._drag_front.lons[self._drag_vertex_idx] = lon
        self._drag_front.lats[self._drag_vertex_idx] = lat

        # Blitting: restaurar fondo y redibujar frente
        self.mw.map_widget.restore_background()
        self._draw_temp_front(self._drag_front)
        self._draw_vertex_markers(self._drag_front)
        self.mw.map_widget.blit()

    # --- Mode handlers ---

    def _handle_select(self, event, lon, lat):
        """Selecciona el frente mas cercano al click."""
        front_id = self._find_nearest_front(event)
        if front_id:
            self.selected_front_id = front_id
            self.mw.statusbar.showMessage(
                f"Frente seleccionado: {front_id}", 3000
            )
            # Actualizar combo de tipo
            front = self.mw.fronts.get_by_id(front_id)
            if front:
                type_names = {
                    FrontType.COLD: "Frio",
                    FrontType.WARM: "Calido",
                    FrontType.OCCLUDED: "Ocluido",
                    FrontType.STATIONARY: "Estacionario",
                }
                self.mw.type_combo.blockSignals(True)
                self.mw.type_combo.setCurrentText(
                    type_names.get(front.front_type, "Frio")
                )
                self.mw.type_combo.blockSignals(False)
        else:
            self.selected_front_id = None
        self.mw._refresh_map()

    def _handle_drag_start(self, event, lon, lat):
        """Inicia el arrastre de un vertice."""
        if not self.selected_front_id:
            return

        front = self.mw.fronts.get_by_id(self.selected_front_id)
        if front is None:
            return

        # Encontrar el vertice mas cercano
        vertex_idx = self._find_nearest_vertex(event, front)
        if vertex_idx is not None:
            self.mw._push_undo()
            self._dragging = True
            self._drag_vertex_idx = vertex_idx
            self._drag_front = front
            # Cachear fondo para blitting
            self.mw.map_widget.cache_background()

    def _handle_add_point(self, event, lon, lat):
        """Agrega un punto al frente en construccion."""
        if event.dblclick:
            # Doble click: finalizar frente
            self._finalize_add()
            return

        self._adding_points_lons.append(lon)
        self._adding_points_lats.append(lat)

        # Actualizar visualizacion temporal
        self._update_add_preview()

    def _handle_delete(self, event, lon, lat):
        """Borra el frente mas cercano al click."""
        front_id = self._find_nearest_front(event)
        if front_id:
            self.mw._push_undo()
            self.mw.fronts.remove(front_id)
            self.mw.statusbar.showMessage(
                f"Frente eliminado: {front_id}", 3000
            )
            self.mw._refresh_map()

    def _finalize_add(self):
        """Finaliza la creacion de un nuevo frente."""
        if len(self._adding_points_lons) >= 2:
            self.mw._push_undo()
            front = Front(
                front_type=self.current_front_type,
                lats=np.array(self._adding_points_lats),
                lons=np.array(self._adding_points_lons),
            )
            self.mw.fronts.add(front)
            self.mw.statusbar.showMessage(
                f"Frente creado: {front.id} ({front.npoints} puntos)", 3000
            )

        self._adding_points_lons.clear()
        self._adding_points_lats.clear()
        self._clear_temp_artists()
        self.mw._refresh_map()

    # --- Helpers ---

    def _find_nearest_front(self, event) -> str | None:
        """Encuentra el frente mas cercano al click en coordenadas de pixel."""
        if not self.mw.fronts:
            return None

        min_dist = float("inf")
        best_id = None
        transform = ccrs.PlateCarree()._as_mpl_transform(self.ax)

        for front in self.mw.fronts:
            for flon, flat in zip(front.lons, front.lats):
                # Convertir lon/lat a coordenadas de display
                px, py = transform.transform_point((flon, flat))
                dist = np.sqrt((px - event.x) ** 2 + (py - event.y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_id = front.id

        if min_dist < PICK_TOLERANCE_PX:
            return best_id
        return None

    def _find_nearest_vertex(self, event, front: Front) -> int | None:
        """Encuentra el vertice mas cercano del frente dado."""
        transform = ccrs.PlateCarree()._as_mpl_transform(self.ax)
        min_dist = float("inf")
        best_idx = None

        for i, (flon, flat) in enumerate(zip(front.lons, front.lats)):
            px, py = transform.transform_point((flon, flat))
            dist = np.sqrt((px - event.x) ** 2 + (py - event.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        if min_dist < PICK_TOLERANCE_PX * 2:
            return best_idx
        return None

    def _draw_temp_front(self, front: Front):
        """Dibuja un frente temporal durante el arrastre."""
        from mapa_frentes.plotting.front_renderer import FRONT_STYLES
        style = FRONT_STYLES.get(front.front_type, FRONT_STYLES[FrontType.COLD])

        if self._highlight_line is not None:
            self._highlight_line.remove()
        self._highlight_line, = self.ax.plot(
            front.lons, front.lats,
            color=style["color"],
            linewidth=self.mw.cfg.plotting.front_linewidth,
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        self.ax.draw_artist(self._highlight_line)

    def _draw_vertex_markers(self, front: Front):
        """Dibuja los marcadores de vertices."""
        if self._vertex_markers is not None:
            self._vertex_markers.remove()
        self._vertex_markers, = self.ax.plot(
            front.lons, front.lats,
            "ko",
            markersize=6,
            transform=ccrs.PlateCarree(),
            zorder=11,
        )
        self.ax.draw_artist(self._vertex_markers)

    def _update_add_preview(self):
        """Actualiza la preview del frente en construccion."""
        if self._add_line is not None:
            self._add_line.remove()
        if self._add_markers is not None:
            self._add_markers.remove()

        if len(self._adding_points_lons) > 0:
            self._add_markers, = self.ax.plot(
                self._adding_points_lons, self._adding_points_lats,
                "ro", markersize=5,
                transform=ccrs.PlateCarree(), zorder=10,
            )
        if len(self._adding_points_lons) > 1:
            self._add_line, = self.ax.plot(
                self._adding_points_lons, self._adding_points_lats,
                "r-", linewidth=2,
                transform=ccrs.PlateCarree(), zorder=10,
            )
        self.canvas.draw_idle()

    def _clear_temp_artists(self):
        """Elimina artistas temporales."""
        for artist in (
            self._highlight_line, self._vertex_markers,
            self._add_line, self._add_markers,
        ):
            if artist is not None:
                try:
                    artist.remove()
                except ValueError:
                    pass
        self._highlight_line = None
        self._vertex_markers = None
        self._add_line = None
        self._add_markers = None
