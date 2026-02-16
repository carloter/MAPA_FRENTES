"""Ventana principal de la aplicacion MAPA_FRENTES."""

import logging
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QAction, QStatusBar, QToolBar,
    QFileDialog, QMessageBox, QComboBox, QLabel,
    QProgressBar, QApplication, QInputDialog, QCheckBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence

from mapa_frentes.config import AppConfig, load_config
from mapa_frentes.fronts.models import FrontCollection, FrontType
from mapa_frentes.gui.map_widget import MapWidget
from mapa_frentes.gui.dialogs import DateSelectorDialog, ConfigDialog, MosaicConfigDialog

logger = logging.getLogger(__name__)


class DataWorker(QThread):
    """Worker thread para descarga y procesamiento de datos."""
    finished = pyqtSignal(object)  # emite el Dataset
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, cfg, date=None, step=None):
        super().__init__()
        self.cfg = cfg
        self.date = date
        self.step = step

    def run(self):
        try:
            from mapa_frentes.data.ecmwf_download import download_ecmwf
            from mapa_frentes.data.grib_reader import read_grib_files

            step_info = f" T+{self.step}h" if self.step else ""
            self.progress.emit(f"Descargando datos ECMWF{step_info}...")
            grib_paths = download_ecmwf(
                self.cfg, date=self.date, step=self.step,
            )

            self.progress.emit("Leyendo datos GRIB2...")
            ds = read_grib_files(grib_paths, self.cfg)

            self.finished.emit(ds)
        except Exception as e:
            self.error.emit(str(e))


class TemporalWorker(QThread):
    """Worker thread para deteccion temporal de frentes (multi-step)."""
    finished = pyqtSignal(object)  # emite FrontCollection
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, cfg, date=None, step=None):
        super().__init__()
        self.cfg = cfg
        self.date = date
        self.step = step or 0

    def run(self):
        try:
            from mapa_frentes.fronts.temporal import compute_temporal_fronts
            fronts = compute_temporal_fronts(
                self.cfg, date=self.date, base_step=self.step,
            )
            self.finished.emit(fronts)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Ventana principal con toolbar, mapa, y barra de estado."""

    def __init__(self, cfg: AppConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        self.ds = None                     # Dataset xarray actual
        self.fronts = FrontCollection()    # Frentes actuales
        self.centers = []                  # Centros de presion (persistentes para nombrado)
        self._undo_stack = []              # Pila de undo
        self._redo_stack = []              # Pila de redo
        self._editor = None                # FrontEditor (creado despues)

        self.setWindowTitle("MAPA FRENTES - MeteoGalicia")
        self.setMinimumSize(1200, 800)

        self._setup_map()
        self._setup_toolbar()
        self._setup_menubar()
        self._setup_statusbar()

    def _setup_map(self):
        self.map_widget = MapWidget(self.cfg, parent=self)
        self.setCentralWidget(self.map_widget)

        # Guardar la posicion ORIGINAL del axes (antes de cualquier colorbar)
        self._original_ax_pos = self.map_widget.ax.get_position()

    def _setup_toolbar(self):
        toolbar = QToolBar("Herramientas")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Descargar datos
        self.act_download = QAction("Descargar", self)
        self.act_download.setToolTip("Descargar datos ECMWF")
        self.act_download.triggered.connect(self._on_download)
        toolbar.addAction(self.act_download)

        toolbar.addSeparator()

        # Detectar frentes
        self.act_detect = QAction("Detectar frentes", self)
        self.act_detect.setToolTip("Ejecutar deteccion automatica TFP")
        self.act_detect.triggered.connect(self._on_detect_fronts)
        self.act_detect.setEnabled(False)
        toolbar.addAction(self.act_detect)

        self.temporal_check = QCheckBox("Temporal")
        self.temporal_check.setToolTip(
            "Usar secuencia temporal para filtrado y clasificacion"
        )
        toolbar.addWidget(self.temporal_check)

        toolbar.addSeparator()

        # Selector de modo de edicion
        toolbar.addWidget(QLabel(" Modo: "))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Navegar", "Seleccionar", "Arrastrar",
            "Anadir frente", "Borrar frente",
            "Generar desde B",
        ])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self.mode_combo)

        # Tipo de frente para nuevos frentes
        toolbar.addWidget(QLabel(" Tipo: "))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Frio", "Calido", "Ocluido", "Estacionario", "Linea inestabilidad"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        toolbar.addWidget(self.type_combo)

        # Conectar frente a borrasca
        self.act_connect_b = QAction("Conectar a B", self)
        self.act_connect_b.setToolTip(
            "Extiende el frente seleccionado hasta la borrasca mas cercana"
        )
        self.act_connect_b.triggered.connect(self._on_connect_to_b)
        toolbar.addAction(self.act_connect_b)

        toolbar.addSeparator()

        # Checkbox para mostrar lineas de asociacion
        self.assoc_check = QCheckBox("Asociaciones")
        self.assoc_check.setChecked(True)
        self.assoc_check.setToolTip("Mostrar lineas de asociacion frente-borrasca")
        self.assoc_check.toggled.connect(lambda: self._refresh_map())
        toolbar.addWidget(self.assoc_check)

        toolbar.addSeparator()

        # Selector de campo de fondo
        toolbar.addWidget(QLabel(" Fondo: "))
        self.bg_combo = QComboBox()
        self.bg_combo.addItem("Ninguno", "none")
        self.bg_combo.addItem("θe 850 hPa", "theta_e_850")
        self.bg_combo.addItem("θe 700 hPa", "theta_e_700")
        self.bg_combo.addItem("|∇θe| 850 hPa", "grad_theta_e_850")
        self.bg_combo.addItem("|∇T| 850 hPa", "grad_t_850")
        self.bg_combo.addItem("Espesor 1000-500", "thickness_1000_500")
        self.bg_combo.addItem("Adv. T 850 hPa", "temp_advection_850")
        self.bg_combo.addItem("Viento 850 hPa", "wind_speed_850")
        self.bg_combo.addItem("Viento 500 hPa", "wind_speed_500")
        self.bg_combo.addItem("Vorticidad 850 hPa", "vorticity_850")
        self.bg_combo.addItem("T 850 hPa", "temp_850")
        self.bg_combo.addItem("q 850 hPa", "humidity_850")
        self.bg_combo.setToolTip("Campo derivado IFS como fondo del mapa")
        self.bg_combo.currentIndexChanged.connect(self._on_bg_field_changed)
        toolbar.addWidget(self.bg_combo)
        self._cached_bg_field = None   # cache del campo calculado
        self._cached_bg_name = "none"  # nombre del campo en cache
        self._bg_cache = {}            # cache multi-campo para mosaico

        toolbar.addSeparator()

        # Overlays opcionales
        self.precip_check = QCheckBox("Precip")
        self.precip_check.setToolTip("Superponer precipitacion acumulada")
        self.precip_check.toggled.connect(lambda: self._refresh_map())
        toolbar.addWidget(self.precip_check)

        self.wind_850_check = QCheckBox("Viento 850")
        self.wind_850_check.setToolTip("Superponer vectores de viento a 850 hPa")
        self.wind_850_check.toggled.connect(lambda: self._refresh_map())
        toolbar.addWidget(self.wind_850_check)

        self.wind_700_check = QCheckBox("Viento 700")
        self.wind_700_check.setToolTip("Superponer vectores de viento a 700 hPa")
        self.wind_700_check.toggled.connect(lambda: self._refresh_map())
        toolbar.addWidget(self.wind_700_check)

        toolbar.addSeparator()

        # Mosaico
        self.act_mosaic = QAction("Mosaico", self)
        self.act_mosaic.setCheckable(True)
        self.act_mosaic.setToolTip("Alternar vista mosaico multi-panel")
        self.act_mosaic.triggered.connect(self._on_toggle_mosaic)
        toolbar.addAction(self.act_mosaic)
        self._mosaic_fields = None   # lista de campos del mosaico
        self._mosaic_layout = (2, 3)  # nrows, ncols

        toolbar.addSeparator()

        # Exportar
        self.act_export = QAction("Exportar", self)
        self.act_export.setToolTip("Exportar mapa como PNG/PDF")
        self.act_export.triggered.connect(self._on_export)
        toolbar.addAction(self.act_export)

    def _setup_menubar(self):
        menubar = self.menuBar()

        # Menu Archivo
        file_menu = menubar.addMenu("&Archivo")

        act_download = file_menu.addAction("&Descargar datos...")
        act_download.triggered.connect(self._on_download)

        file_menu.addSeparator()

        act_load = file_menu.addAction("&Cargar sesion...")
        act_load.setShortcut(QKeySequence("Ctrl+O"))
        act_load.triggered.connect(self._on_load_session)

        act_save = file_menu.addAction("&Guardar sesion...")
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self._on_save_session)

        file_menu.addSeparator()

        act_export = file_menu.addAction("&Exportar mapa...")
        act_export.setShortcut(QKeySequence("Ctrl+E"))
        act_export.triggered.connect(self._on_export)

        file_menu.addSeparator()

        act_quit = file_menu.addAction("&Salir")
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)

        # Menu Editar
        edit_menu = menubar.addMenu("&Editar")

        self.act_undo = edit_menu.addAction("&Deshacer")
        self.act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self.act_undo.triggered.connect(self._on_undo)
        self.act_undo.setEnabled(False)

        self.act_redo = edit_menu.addAction("&Rehacer")
        self.act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        self.act_redo.triggered.connect(self._on_redo)
        self.act_redo.setEnabled(False)

        edit_menu.addSeparator()

        act_delete = edit_menu.addAction("&Borrar frente seleccionado")
        act_delete.setShortcut(QKeySequence("Delete"))
        act_delete.triggered.connect(self._on_delete_selected)

        edit_menu.addSeparator()

        act_name_storm = edit_menu.addAction("&Nombrar borrasca...")
        act_name_storm.setShortcut(QKeySequence("Ctrl+N"))
        act_name_storm.triggered.connect(self._on_name_storm)

        edit_menu.addSeparator()

        act_connect = edit_menu.addAction("Conectar frente a &B")
        act_connect.setShortcut(QKeySequence("Ctrl+B"))
        act_connect.triggered.connect(self._on_connect_to_b)

        # Menu Analisis
        analysis_menu = menubar.addMenu("&Analisis")

        act_detect = analysis_menu.addAction("&Detectar frentes")
        act_detect.triggered.connect(self._on_detect_fronts)

        act_gen_center = analysis_menu.addAction("&Generar frentes desde borrasca")
        act_gen_center.setToolTip(
            "Seleccione modo 'Generar desde B' y haga click en un centro L"
        )
        act_gen_center.triggered.connect(self._on_activate_generate_mode)

        # Menu Configuracion
        config_menu = menubar.addMenu("&Configuracion")

        act_config = config_menu.addAction("&Parametros...")
        act_config.triggered.connect(self._on_config)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progress_bar)

        self.coord_label = QLabel("")
        self.statusbar.addPermanentWidget(self.coord_label)

        # Conectar evento de movimiento del raton para mostrar coordenadas
        self.map_widget.canvas.mpl_connect(
            "motion_notify_event", self._on_mouse_move
        )

    def _on_mouse_move(self, event):
        """Muestra coordenadas lon/lat en la barra de estado."""
        if event.inaxes is not None and event.inaxes in self.map_widget.get_valid_axes():
            try:
                import cartopy.crs as ccrs
                lon, lat = ccrs.PlateCarree().transform_point(
                    event.xdata, event.ydata, event.inaxes.projection,
                )
                self.coord_label.setText(
                    f"Lon: {lon:.2f}  Lat: {lat:.2f}"
                )
            except Exception:
                self.coord_label.setText("")
        else:
            self.coord_label.setText("")

    # --- Acciones ---

    def _on_download(self):
        """Abre dialogo de fecha y descarga datos."""
        dlg = DateSelectorDialog(self)
        if dlg.exec_() != dlg.Accepted:
            return

        date = dlg.get_datetime()
        step = dlg.get_step()
        self._start_download(date, step)

    def _start_download(self, date=None, step=None):
        """Inicia la descarga en un thread secundario."""
        self.act_download.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminado

        self._worker = DataWorker(self.cfg, date=date, step=step)
        self._worker.finished.connect(self._on_download_finished)
        self._worker.error.connect(self._on_download_error)
        self._worker.progress.connect(
            lambda msg: self.statusbar.showMessage(msg)
        )
        self._worker.start()

    def _on_download_finished(self, ds):
        """Callback cuando la descarga termina."""
        self.ds = ds
        self._cached_bg_field = None  # invalidar cache de campo de fondo
        self._cached_bg_name = "none"
        self._bg_cache.clear()        # invalidar cache mosaico
        self.act_download.setEnabled(True)
        self.act_detect.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusbar.showMessage("Datos descargados correctamente", 5000)

        # Recalcular centros de presion
        self._recompute_centers()
        self._refresh_map()

    def _on_download_error(self, error_msg):
        """Callback cuando la descarga falla."""
        self.act_download.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusbar.showMessage("Error en descarga", 5000)
        QMessageBox.critical(self, "Error de descarga", error_msg)

    def _on_detect_fronts(self):
        """Ejecuta deteccion automatica de frentes."""
        if self.ds is None:
            QMessageBox.warning(
                self, "Sin datos",
                "Primero descargue datos ECMWF."
            )
            return

        if self.temporal_check.isChecked():
            self._on_detect_fronts_temporal()
            return

        self.statusbar.showMessage("Detectando frentes...")
        QApplication.processEvents()

        try:
            from mapa_frentes.fronts.tfp import compute_tfp_fronts
            from mapa_frentes.fronts.classifier import classify_fronts
            from mapa_frentes.fronts.instability import detect_instability_lines

            self._push_undo()
            self.fronts = compute_tfp_fronts(self.ds, self.cfg)

            # Asociar a centros ANTES de clasificar (el clasificador necesita los centros)
            if self.centers:
                from mapa_frentes.fronts.association import associate_fronts_to_centers
                self.fronts = associate_fronts_to_centers(
                    self.fronts, self.centers, self.cfg,
                )
            self.fronts = classify_fronts(
                self.fronts, self.ds, self.cfg, centers=self.centers
            )

            # Lineas de inestabilidad
            instab_lines = detect_instability_lines(self.ds, self.cfg)
            for il in instab_lines:
                self.fronts.add(il)

            self.statusbar.showMessage(
                f"Frentes detectados: {len(self.fronts)}", 5000
            )
        except Exception as e:
            logger.error("Error detectando frentes: %s", e)
            QMessageBox.critical(self, "Error", str(e))
            self.statusbar.showMessage("Error en deteccion", 5000)
            return

        self._refresh_map()

    def _on_detect_fronts_temporal(self):
        """Ejecuta deteccion temporal de frentes en un thread."""
        self.act_detect.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.statusbar.showMessage("Detectando frentes (modo temporal)...")

        self._push_undo()
        self._temporal_worker = TemporalWorker(self.cfg)
        self._temporal_worker.finished.connect(self._on_temporal_finished)
        self._temporal_worker.error.connect(self._on_temporal_error)
        self._temporal_worker.start()

    def _on_temporal_finished(self, fronts):
        """Callback cuando la deteccion temporal termina."""
        self.fronts = fronts
        self.act_detect.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusbar.showMessage(
            f"Frentes detectados (temporal): {len(self.fronts)}", 5000
        )
        self._refresh_map()

    def _on_temporal_error(self, error_msg):
        """Callback cuando la deteccion temporal falla."""
        self.act_detect.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusbar.showMessage("Error en deteccion temporal", 5000)
        QMessageBox.critical(self, "Error temporal", error_msg)

    def _on_export(self):
        """Exporta el mapa actual a fichero."""
        default_dir = str(Path(self.cfg.data.output_dir).resolve())
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Exportar mapa", default_dir,
            "PNG (*.png);;PDF (*.pdf)"
        )
        if filepath:
            from mapa_frentes.plotting.export import export_map
            export_map(self.map_widget.fig, filepath, self.cfg)
            self.statusbar.showMessage(f"Exportado: {filepath}", 5000)

    def _on_save_session(self):
        """Guarda la sesion actual (frentes) como GeoJSON."""
        default_dir = str(Path(self.cfg.data.sessions_dir).resolve())
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Guardar sesion", default_dir,
            "GeoJSON (*.geojson)"
        )
        if filepath:
            from mapa_frentes.fronts.io import save_session
            save_session(self.fronts, filepath)
            self.statusbar.showMessage(f"Sesion guardada: {filepath}", 5000)

    def _on_load_session(self):
        """Carga frentes desde fichero GeoJSON."""
        default_dir = str(Path(self.cfg.data.sessions_dir).resolve())
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Cargar sesion", default_dir,
            "GeoJSON (*.geojson)"
        )
        if filepath:
            from mapa_frentes.fronts.io import load_session
            self._push_undo()
            self.fronts = load_session(filepath)
            self._refresh_map()
            self.statusbar.showMessage(
                f"Sesion cargada: {len(self.fronts)} frentes", 5000
            )

    def _on_config(self):
        """Abre el dialogo de configuracion."""
        dlg = ConfigDialog(self.cfg, self)
        if dlg.exec_() == dlg.Accepted:
            dlg.apply_to_config()
            self.statusbar.showMessage("Configuracion actualizada", 3000)
            if self.ds is not None:
                self._refresh_map()

    def _on_activate_generate_mode(self):
        """Activa el modo de generacion desde borrascas."""
        if self.ds is None:
            QMessageBox.warning(
                self, "Sin datos",
                "Primero descargue datos ECMWF.",
            )
            return
        if not self.centers or not any(c.type == "L" for c in self.centers):
            QMessageBox.warning(
                self, "Sin borrascas",
                "No hay borrascas detectadas. Descargue datos primero.",
            )
            return
        self.mode_combo.setCurrentText("Generar desde B")

    def _on_mode_changed(self, mode_text: str):
        """Cambia el modo de edicion."""
        if self._editor is not None:
            mode_map = {
                "Navegar": "navigate",
                "Seleccionar": "select",
                "Arrastrar": "drag",
                "Anadir frente": "add",
                "Borrar frente": "delete",
                "Generar desde B": "generate",
            }
            self._editor.set_mode(mode_map.get(mode_text, "navigate"))

        # Desactivar toolbar de Matplotlib en modos de edicion
        if mode_text == "Navegar":
            self.map_widget.toolbar.setEnabled(True)
        else:
            self.map_widget.toolbar.setEnabled(False)

    def _on_type_changed(self, type_text: str):
        """Cambia el tipo de frente para nuevos frentes o el seleccionado."""
        type_map = {
            "Frio": FrontType.COLD,
            "Calido": FrontType.WARM,
            "Ocluido": FrontType.OCCLUDED,
            "Estacionario": FrontType.STATIONARY,
            "Linea inestabilidad": FrontType.INSTABILITY_LINE,
        }
        new_type = type_map.get(type_text, FrontType.COLD)

        if self._editor is not None:
            self._editor.current_front_type = new_type
            # Si hay un frente seleccionado, cambiar su tipo
            if self._editor.selected_front_id:
                front = self.fronts.get_by_id(self._editor.selected_front_id)
                if front is not None:
                    self._push_undo()
                    front.front_type = new_type
                    self._refresh_map()

    def _on_bg_field_changed(self):
        """Cambia el campo de fondo y refresca el mapa."""
        self._cached_bg_field = None  # invalidar cache
        self._cached_bg_name = "none"
        self._refresh_map()

    def _on_toggle_mosaic(self, checked):
        """Alterna entre vista unica y mosaico."""
        if checked:
            from mapa_frentes.analysis.derived_fields import AVAILABLE_FIELDS
            dlg = MosaicConfigDialog(
                AVAILABLE_FIELDS,
                current_selection=self._mosaic_fields,
                parent=self,
            )
            if dlg.exec_() != dlg.Accepted:
                self.act_mosaic.setChecked(False)
                return
            self._mosaic_fields = dlg.get_field_names()
            self._mosaic_layout = dlg.get_layout()
            nrows, ncols = self._mosaic_layout
            self.map_widget.switch_to_mosaic(
                self._mosaic_fields, nrows, ncols,
            )
            self._bg_cache.clear()
            self._refresh_mosaic()
        else:
            self.map_widget.switch_to_single()
            self._original_ax_pos = self.map_widget.ax.get_position()
            self._refresh_map()

    def _get_bg_field(self, field_name, lats, lons):
        """Obtiene campo de fondo con cache para evitar recalcular."""
        if self._cached_bg_name == field_name and self._cached_bg_field is not None:
            return self._cached_bg_field

        from mapa_frentes.analysis.derived_fields import compute_derived_field
        derived = compute_derived_field(self.ds, field_name, lats, lons)
        if derived is not None:
            self._cached_bg_field = derived
            self._cached_bg_name = field_name
        return derived

    def _on_delete_selected(self):
        """Borra el frente seleccionado."""
        if self._editor is not None and self._editor.selected_front_id:
            self._push_undo()
            self.fronts.remove(self._editor.selected_front_id)
            self._editor.selected_front_id = None
            self._refresh_map()

    def _on_connect_to_b(self):
        """Extiende el frente seleccionado hasta la borrasca mas cercana."""
        if self._editor is None or not self._editor.selected_front_id:
            self.statusbar.showMessage(
                "Seleccione un frente primero", 3000,
            )
            return

        front = self.fronts.get_by_id(self._editor.selected_front_id)
        if front is None:
            return

        lows = [c for c in self.centers if c.type == "L"]
        if not lows:
            self.statusbar.showMessage("No hay borrascas detectadas", 3000)
            return

        from mapa_frentes.fronts.association import (
            find_nearest_center_for_front, smooth_extend_to_center,
        )

        center, which_end, dist = find_nearest_center_for_front(front, lows)
        max_dist = self.cfg.center_fronts.max_association_distance_deg

        if center is None or dist > max_dist:
            self.statusbar.showMessage(
                f"No hay borrasca dentro del umbral ({max_dist:.0f} deg)", 3000,
            )
            return

        self._push_undo()
        smooth_extend_to_center(front, center, which_end, self.cfg)
        self.statusbar.showMessage(
            f"Frente conectado a {center.id} ({center.value:.0f} hPa)", 5000,
        )
        self._refresh_map()

    # --- Undo/Redo ---

    def _push_undo(self):
        """Guarda el estado actual de los frentes para undo."""
        import copy
        self._undo_stack.append(copy.deepcopy(self.fronts))
        self._redo_stack.clear()
        self.act_undo.setEnabled(True)
        self.act_redo.setEnabled(False)

    def _on_undo(self):
        """Deshace el ultimo cambio en frentes."""
        if not self._undo_stack:
            return
        import copy
        self._redo_stack.append(copy.deepcopy(self.fronts))
        self.fronts = self._undo_stack.pop()
        self.act_undo.setEnabled(bool(self._undo_stack))
        self.act_redo.setEnabled(True)
        self._refresh_map()

    def _on_redo(self):
        """Rehace el ultimo cambio deshecho."""
        if not self._redo_stack:
            return
        import copy
        self._undo_stack.append(copy.deepcopy(self.fronts))
        self.fronts = self._redo_stack.pop()
        self.act_undo.setEnabled(True)
        self.act_redo.setEnabled(bool(self._redo_stack))
        self._refresh_map()

    # --- Renderizado ---

    def _refresh_map(self):
        """Redibuja el mapa completo con isobaras y frentes."""
        if self.map_widget.mosaic_mode:
            self._refresh_mosaic()
            return

        ax = self.map_widget.ax
        # Limpiar artistas previos (mantener fondo cartopy)
        # Guardamos la extent actual antes de limpiar
        try:
            current_extent = ax.get_extent()
        except Exception:
            current_extent = None

        ax.clear()

        # Eliminar colorbars previas y restaurar geometria del axes principal
        
        fig = ax.get_figure()

        # Borra axes extra (típicamente colorbars creadas como axes separados)
        while len(fig.axes) > 1:
            fig.delaxes(fig.axes[-1])

        # Restaurar SIEMPRE la geometria original del axes principal
        ax.set_position(self._original_ax_pos)


        # Re-aplicar decoracion base
        from mapa_frentes.plotting.map_canvas import create_map_figure
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        area = self.cfg.area
        plot_cfg = self.cfg.plotting

        ax.set_extent(
            [area.lon_min, area.lon_max, area.lat_min, area.lat_max],
            crs=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.OCEAN, facecolor=plot_cfg.ocean_color, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=plot_cfg.land_color, zorder=0)
        ax.add_feature(
            cfeature.COASTLINE,
            edgecolor=plot_cfg.coastline_color,
            linewidth=plot_cfg.coastline_linewidth,
            zorder=1,
        )
        ax.add_feature(
            cfeature.BORDERS,
            edgecolor=plot_cfg.border_color,
            linewidth=plot_cfg.border_linewidth,
            zorder=1,
        )
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                          alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

        # Tamaño de etiquetas de la rejilla
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}

        # Dibujar datos si disponibles
        if self.ds is not None:
            from mapa_frentes.analysis.isobars import (
                smooth_mslp, compute_isobar_levels,
            )
            from mapa_frentes.plotting.isobar_renderer import (
                draw_isobars, draw_pressure_labels, draw_background_field,
            )

            lat_name = "latitude" if "latitude" in self.ds.coords else "lat"
            lon_name = "longitude" if "longitude" in self.ds.coords else "lon"
            lats = self.ds[lat_name].values
            lons = self.ds[lon_name].values

            # Campo de fondo derivado (antes de isobaras)
            bg_name = self.bg_combo.currentData()
            if bg_name and bg_name != "none":
                derived = self._get_bg_field(bg_name, lats, lons)
                if derived is not None:
                    draw_background_field(ax, derived, lons, lats, self.cfg)

            msl_smooth = smooth_mslp(
                self.ds["msl"], sigma=self.cfg.isobars.smooth_sigma
            )
            levels = compute_isobar_levels(
                msl_smooth, interval=self.cfg.isobars.interval_hpa
            )

            draw_isobars(ax, msl_smooth, lons, lats, levels, self.cfg)
            draw_pressure_labels(ax, self.centers, self.cfg)

            # Overlays opcionales
            if self.precip_check.isChecked():
                from mapa_frentes.plotting.isobar_renderer import draw_precipitation
                draw_precipitation(ax, self.ds, lons, lats, self.cfg)
            if self.wind_850_check.isChecked():
                from mapa_frentes.plotting.isobar_renderer import draw_wind_vectors
                draw_wind_vectors(ax, self.ds, lons, lats, 850, self.cfg)
            if self.wind_700_check.isChecked():
                from mapa_frentes.plotting.isobar_renderer import draw_wind_vectors
                draw_wind_vectors(ax, self.ds, lons, lats, 700, self.cfg)

            # Titulo
            time_str = ""
            for coord_name in ("time", "valid_time"):
                if coord_name in self.ds.coords:
                    time_str = str(self.ds.coords[coord_name].values)[:16]
                    break
            ax.set_title(
                f"Analisis en superficie - MSLP (hPa)\n{time_str}",
                fontsize=12, fontweight="bold",
            )

        # Dibujar frentes
        if self.fronts and len(self.fronts) > 0:
            from mapa_frentes.plotting.front_renderer import (
                draw_fronts, draw_front_legend,
            )
            highlight = None
            if self._editor is not None:
                highlight = self._editor.selected_front_id
            draw_fronts(ax, self.fronts, self.cfg, highlight_id=highlight)
            draw_front_legend(ax, self.cfg)

            # Lineas de asociacion frente-borrasca
            if self.assoc_check.isChecked() and self.centers:
                from mapa_frentes.plotting.front_renderer import (
                    draw_association_lines,
                )
                draw_association_lines(
                    ax, self.fronts, self.centers, self.cfg,
                )

        # Restaurar extent si estabamos con zoom
        if current_extent is not None:
            try:
                ax.set_extent(current_extent, crs=ccrs.PlateCarree())
            except Exception:
                pass
        # Dejar espacio inferior para la leyenda
        ax.figure.subplots_adjust(bottom=0.10)
        self.map_widget.redraw()

    def _refresh_mosaic(self):
        """Redibuja todos los paneles del mosaico."""
        from mapa_frentes.plotting.map_canvas import apply_base_cartography
        from mapa_frentes.analysis.derived_fields import AVAILABLE_FIELDS

        msl_smooth = None
        levels = None
        lats = lons = None

        if self.ds is not None:
            from mapa_frentes.analysis.isobars import (
                smooth_mslp, compute_isobar_levels,
            )
            from mapa_frentes.plotting.isobar_renderer import (
                draw_isobars, draw_pressure_labels, draw_background_field,
            )

            lat_name = "latitude" if "latitude" in self.ds.coords else "lat"
            lon_name = "longitude" if "longitude" in self.ds.coords else "lon"
            lats = self.ds[lat_name].values
            lons = self.ds[lon_name].values

            msl_smooth = smooth_mslp(
                self.ds["msl"], sigma=self.cfg.isobars.smooth_sigma,
            )
            levels = compute_isobar_levels(
                msl_smooth, interval=self.cfg.isobars.interval_hpa,
            )

        highlight = None
        if self._editor is not None:
            highlight = self._editor.selected_front_id

        for field_name, ax in self.map_widget.axes_dict.items():
            ax.clear()
            apply_base_cartography(ax, self.cfg, lightweight=True)

            if self.ds is not None and lats is not None:
                # Campo de fondo
                if field_name != "none":
                    derived = self._get_bg_field_cached(field_name, lats, lons)
                    if derived is not None:
                        draw_background_field(ax, derived, lons, lats, self.cfg)

                # Isobaras
                draw_isobars(ax, msl_smooth, lons, lats, levels, self.cfg)
                draw_pressure_labels(ax, self.centers, self.cfg)

                # Overlays opcionales
                if self.precip_check.isChecked():
                    from mapa_frentes.plotting.isobar_renderer import draw_precipitation
                    draw_precipitation(ax, self.ds, lons, lats, self.cfg)
                if self.wind_850_check.isChecked():
                    from mapa_frentes.plotting.isobar_renderer import draw_wind_vectors
                    draw_wind_vectors(ax, self.ds, lons, lats, 850, self.cfg)
                if self.wind_700_check.isChecked():
                    from mapa_frentes.plotting.isobar_renderer import draw_wind_vectors
                    draw_wind_vectors(ax, self.ds, lons, lats, 700, self.cfg)

            # Frentes (mismos en todos los paneles)
            if self.fronts and len(self.fronts) > 0:
                from mapa_frentes.plotting.front_renderer import draw_fronts
                draw_fronts(ax, self.fronts, self.cfg, highlight_id=highlight)

                if self.assoc_check.isChecked() and self.centers:
                    from mapa_frentes.plotting.front_renderer import (
                        draw_association_lines,
                    )
                    draw_association_lines(
                        ax, self.fronts, self.centers, self.cfg,
                    )

            # Titulo del panel
            label = AVAILABLE_FIELDS.get(field_name, field_name)
            ax.set_title(label, fontsize=8, pad=2)

        self.map_widget.redraw()

    def _get_bg_field_cached(self, field_name, lats, lons):
        """Obtiene campo de fondo con cache multi-campo para mosaico."""
        if field_name in self._bg_cache:
            return self._bg_cache[field_name]
        from mapa_frentes.analysis.derived_fields import compute_derived_field
        derived = compute_derived_field(self.ds, field_name, lats, lons)
        if derived is not None:
            self._bg_cache[field_name] = derived
        return derived

    # --- Centros de presion ---

    def _recompute_centers(self):
        """Recalcula los centros de presion desde los datos."""
        if self.ds is None:
            return
        from mapa_frentes.analysis.isobars import smooth_mslp
        from mapa_frentes.analysis.pressure_centers import detect_pressure_centers

        lat_name = "latitude" if "latitude" in self.ds.coords else "lat"
        lon_name = "longitude" if "longitude" in self.ds.coords else "lon"
        lats = self.ds[lat_name].values
        lons = self.ds[lon_name].values

        msl_smooth = smooth_mslp(
            self.ds["msl"], sigma=self.cfg.isobars.smooth_sigma
        )
        self.centers = detect_pressure_centers(msl_smooth, lats, lons, self.cfg)

    def _on_name_storm(self):
        """Abre dialogo para nombrar la borrasca mas cercana.

        Muestra lista de borrascas (L) detectadas para que el usuario
        elija cual nombrar, luego pide el nombre.
        """
        lows = [c for c in self.centers if c.type == "L"]
        if not lows:
            QMessageBox.information(
                self, "Sin borrascas",
                "No hay borrascas detectadas. Descargue datos primero."
            )
            return

        # Crear lista de opciones
        items = []
        for i, low in enumerate(lows):
            rank = "B" if low.primary else "b"
            name_part = f' "{low.name}"' if low.name else ""
            items.append(
                f"{rank} ({low.value:.0f} hPa) en {low.lat:.1f}N {abs(low.lon):.1f}"
                f"{'W' if low.lon < 0 else 'E'}{name_part}"
            )

        item, ok = QInputDialog.getItem(
            self, "Nombrar borrasca",
            "Seleccione la borrasca:",
            items, 0, False,
        )
        if not ok:
            return

        idx = items.index(item)
        selected_low = lows[idx]

        name, ok = QInputDialog.getText(
            self, "Nombre de borrasca",
            f"Nombre para la borrasca en {selected_low.lat:.1f}N "
            f"{abs(selected_low.lon):.1f}{'W' if selected_low.lon < 0 else 'E'}:",
            text=selected_low.name,
        )
        if ok:
            selected_low.name = name.strip()
            self._refresh_map()
            self.statusbar.showMessage(
                f"Borrasca nombrada: {name}", 5000
            )

    def set_editor(self, editor):
        """Registra el editor de frentes."""
        self._editor = editor
