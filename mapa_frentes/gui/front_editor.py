"""Edicion interactiva de frentes meteorologicos.

Modos de edicion:
- navigate: Navegacion normal (zoom, pan via toolbar Matplotlib)
- select: Click en un frente para seleccionarlo
- drag: Mover vertices del frente seleccionado
- add: Click para colocar puntos, doble-click para finalizar
        Si hay un frente seleccionado y el click esta cerca de un extremo,
        activa modo extension (concatena puntos al frente existente).
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
        self._active_ax = None  # axes donde se hizo el ultimo click

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

        # Estado para extension de frente existente
        self._extending_front = None
        self._extending_end = None

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
            self._extending_front = None
            self._extending_end = None
        self.mode = mode
        self._dragging = False

        if mode in ("navigate",):
            self._clear_temp_artists()

        logger.debug("Modo de edicion: %s", mode)

    def _event_to_lonlat(self, event):
        """Convierte coordenadas del evento (en la proyeccion del axes) a lon/lat."""
        ax = event.inaxes if event.inaxes is not None else self.ax
        return ccrs.PlateCarree().transform_point(
            event.xdata, event.ydata, ax.projection
        )

    def _is_valid_axes(self, event):
        """Comprueba si el evento es en uno de nuestros axes de mapa."""
        if event.inaxes is None:
            return False
        return event.inaxes in self.mw.map_widget.get_valid_axes()

    # --- Event handlers ---

    def _on_press(self, event):
        if not self._is_valid_axes(event) or self.mode == "navigate":
            return
        self._active_ax = event.inaxes
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
        elif self.mode == "generate":
            self._handle_generate(event, lon, lat)

    def _on_release(self, event):
        if self._dragging:
            self._dragging = False
            self._drag_vertex_idx = None
            # Redibujar completamente despues del arrastre
            self.mw._refresh_map()

    def _on_motion(self, event):
        if not self._dragging or not self._is_valid_axes(event):
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
                    FrontType.INSTABILITY_LINE: "Linea inestabilidad",
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
        """Agrega un punto al frente en construccion o activa extension."""
        if event.dblclick:
            # Doble click: finalizar frente
            self._finalize_add()
            return

        # Primer click: comprobar si se extiende un frente existente
        if (
            len(self._adding_points_lons) == 0
            and self._extending_front is None
            and self.selected_front_id
        ):
            front = self.mw.fronts.get_by_id(self.selected_front_id)
            if front is not None:
                near_end = self._check_near_endpoint(event, front)
                if near_end is not None:
                    self._extending_front = front
                    self._extending_end = near_end
                    self.mw.statusbar.showMessage(
                        f"Extendiendo frente {front.id} desde {near_end}",
                        5000,
                    )
                    # Agregar el punto del click como primer punto de extension
                    self._adding_points_lons.append(lon)
                    self._adding_points_lats.append(lat)
                    self._update_add_preview()
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
        """Finaliza la creacion o extension de un frente."""
        if self._extending_front is not None:
            # Modo extension: concatenar puntos al frente existente
            if len(self._adding_points_lons) >= 1:
                self.mw._push_undo()
                new_lats = np.array(self._adding_points_lats)
                new_lons = np.array(self._adding_points_lons)

                if self._extending_end == "start":
                    # Invertir los nuevos puntos y prepend
                    self._extending_front.lats = np.concatenate(
                        [new_lats[::-1], self._extending_front.lats]
                    )
                    self._extending_front.lons = np.concatenate(
                        [new_lons[::-1], self._extending_front.lons]
                    )
                else:
                    # Append
                    self._extending_front.lats = np.concatenate(
                        [self._extending_front.lats, new_lats]
                    )
                    self._extending_front.lons = np.concatenate(
                        [self._extending_front.lons, new_lons]
                    )

                # Comprobar si el nuevo extremo esta cerca de un centro L
                self._try_assign_center_to_extended_front()

                self.mw.statusbar.showMessage(
                    f"Frente extendido: {self._extending_front.id}", 3000,
                )
        elif len(self._adding_points_lons) >= 2:
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
        self._extending_front = None
        self._extending_end = None
        self._clear_temp_artists()
        self.mw._refresh_map()

    def _try_assign_center_to_extended_front(self):
        """Intenta asignar un centro L al frente extendido si el nuevo extremo esta cerca."""
        front = self._extending_front
        if front is None:
            return

        lows = [c for c in getattr(self.mw, "centers", []) if c.type == "L"]
        if not lows:
            return

        from mapa_frentes.fronts.association import find_nearest_center_for_front
        center, which_end, dist = find_nearest_center_for_front(front, lows)

        max_dist = self.mw.cfg.center_fronts.max_association_distance_deg
        if dist <= max_dist and center is not None:
            front.center_id = center.id
            front.association_end = which_end

    # --- Helpers ---

    def _check_near_endpoint(self, event, front: Front) -> str | None:
        """Comprueba si el click esta cerca de un extremo del frente.

        Returns "start", "end" o None.
        """
        ax = self._active_ax or self.ax
        transform = ccrs.PlateCarree()._as_mpl_transform(ax)
        tolerance = PICK_TOLERANCE_PX * 2

        # Extremo inicio
        px_s, py_s = transform.transform_point(
            (front.lons[0], front.lats[0])
        )
        d_start = np.sqrt((px_s - event.x) ** 2 + (py_s - event.y) ** 2)

        # Extremo fin
        px_e, py_e = transform.transform_point(
            (front.lons[-1], front.lats[-1])
        )
        d_end = np.sqrt((px_e - event.x) ** 2 + (py_e - event.y) ** 2)

        if d_start < d_end and d_start < tolerance:
            return "start"
        if d_end <= d_start and d_end < tolerance:
            return "end"
        return None

    def _find_nearest_front(self, event) -> str | None:
        """Encuentra el frente mas cercano al click en coordenadas de pixel."""
        if not self.mw.fronts:
            return None

        min_dist = float("inf")
        best_id = None
        ax = self._active_ax or self.ax
        transform = ccrs.PlateCarree()._as_mpl_transform(ax)

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
        ax = self._active_ax or self.ax
        transform = ccrs.PlateCarree()._as_mpl_transform(ax)
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

        ax = self._active_ax or self.ax
        if self._highlight_line is not None:
            self._highlight_line.remove()
        self._highlight_line, = ax.plot(
            front.lons, front.lats,
            color=style["color"],
            linewidth=self.mw.cfg.plotting.front_linewidth,
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        ax.draw_artist(self._highlight_line)

    def _draw_vertex_markers(self, front: Front):
        """Dibuja los marcadores de vertices."""
        ax = self._active_ax or self.ax
        if self._vertex_markers is not None:
            self._vertex_markers.remove()
        self._vertex_markers, = ax.plot(
            front.lons, front.lats,
            "ko",
            markersize=6,
            transform=ccrs.PlateCarree(),
            zorder=11,
        )
        ax.draw_artist(self._vertex_markers)

    def _update_add_preview(self):
        """Actualiza la preview del frente en construccion o extension."""
        if self._add_line is not None:
            self._add_line.remove()
            self._add_line = None
        if self._add_markers is not None:
            self._add_markers.remove()
            self._add_markers = None

        extending = self._extending_front is not None
        marker_color = "go" if extending else "ro"
        line_color = "g-" if extending else "r-"

        # En modo extension, incluir los ultimos puntos del frente existente
        preview_lons = list(self._adding_points_lons)
        preview_lats = list(self._adding_points_lats)

        if extending and self._extending_front is not None:
            front = self._extending_front
            n_ctx = min(3, front.npoints)
            if self._extending_end == "end":
                ctx_lons = front.lons[-n_ctx:].tolist()
                ctx_lats = front.lats[-n_ctx:].tolist()
                preview_lons = ctx_lons + preview_lons
                preview_lats = ctx_lats + preview_lats
            else:
                ctx_lons = front.lons[:n_ctx].tolist()[::-1]
                ctx_lats = front.lats[:n_ctx].tolist()[::-1]
                preview_lons = ctx_lons + preview_lons
                preview_lats = ctx_lats + preview_lats

        ax = self._active_ax or self.ax
        if len(self._adding_points_lons) > 0:
            self._add_markers, = ax.plot(
                self._adding_points_lons, self._adding_points_lats,
                marker_color, markersize=5,
                transform=ccrs.PlateCarree(), zorder=10,
            )
        if len(preview_lons) > 1:
            self._add_line, = ax.plot(
                preview_lons, preview_lats,
                line_color, linewidth=2,
                transform=ccrs.PlateCarree(), zorder=10,
            )
        self.canvas.draw_idle()

    def _handle_generate(self, event, lon, lat):
        """Genera frentes desde el centro L mas cercano al click."""
        center = self._find_nearest_l_center(event)
        if center is None:
            self.mw.statusbar.showMessage(
                "No hay borrasca (B) cerca del click", 3000,
            )
            return

        if self.mw.ds is None:
            self.mw.statusbar.showMessage(
                "No hay datos cargados para generar frentes", 3000,
            )
            return

        # Verificar si ya existen frentes de este centro
        existing = [
            f for f in self.mw.fronts
            if f.center_id == center.id
        ]
        if existing:
            from PyQt5.QtWidgets import QMessageBox
            resp = QMessageBox.question(
                self.mw, "Frentes existentes",
                f"Ya existen {len(existing)} frentes del centro {center.id}.\n"
                "Desea reemplazarlos?",
            )
            if resp == QMessageBox.Yes:
                self.mw._push_undo()
                for f in existing:
                    self.mw.fronts.remove(f.id)
            else:
                return
        else:
            self.mw._push_undo()

        from mapa_frentes.fronts.center_fronts import generate_fronts_from_center
        generated = generate_fronts_from_center(center, self.mw.ds, self.mw.cfg)

        for front in generated:
            self.mw.fronts.add(front)

        if generated:
            self.mw.statusbar.showMessage(
                f"Generados {len(generated)} frentes desde {center.id} "
                f"({center.value:.0f} hPa)", 5000,
            )
        else:
            self.mw.statusbar.showMessage(
                f"No se detectaron frentes claros desde {center.id}", 5000,
            )

        self.mw._refresh_map()

    def _find_nearest_l_center(self, event):
        """Encuentra el centro L mas cercano al click en coordenadas de pixel."""
        if not hasattr(self.mw, "centers") or not self.mw.centers:
            return None

        ax = self._active_ax or self.ax
        transform = ccrs.PlateCarree()._as_mpl_transform(ax)
        min_dist = float("inf")
        best_center = None

        for center in self.mw.centers:
            if center.type != "L":
                continue
            try:
                px, py = transform.transform_point((center.lon, center.lat))
                dist = np.sqrt((px - event.x) ** 2 + (py - event.y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_center = center
            except Exception:
                continue

        if min_dist < PICK_TOLERANCE_PX * 3:
            return best_center
        return None

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
