"""Dialogos de la aplicacion: selector de fecha, configuracion, mosaico."""

from datetime import datetime

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QDateTimeEdit, QComboBox, QPushButton, QDialogButtonBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QFileDialog, QMessageBox, QGridLayout, QRadioButton, QButtonGroup,
)
from PyQt5.QtCore import QDateTime, Qt


class DateSelectorDialog(QDialog):
    """Dialogo para seleccionar fecha/hora del analisis ECMWF."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar fecha/hora")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)

        # Selector de fecha (solo fecha, la hora se elige con el combo Run)
        layout.addWidget(QLabel("Fecha del analisis:"))
        self.datetime_edit = QDateTimeEdit()
        self.datetime_edit.setDisplayFormat("yyyy-MM-dd")
        self.datetime_edit.setDateTime(QDateTime.currentDateTime())
        self.datetime_edit.setCalendarPopup(True)
        layout.addWidget(self.datetime_edit)

        # Hora del run (0, 6, 12, 18)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Run (hora UTC):"))
        self.run_combo = QComboBox()
        self.run_combo.addItems(["00Z", "06Z", "12Z", "18Z"])
        # Pre-seleccionar el run mas cercano a la hora actual
        self._preselect_nearest_run()
        h_layout.addWidget(self.run_combo)
        layout.addLayout(h_layout)

        # Paso de prediccion (T+0 a T+144, cada 3h)
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Prediccion:"))
        self.step_combo = QComboBox()
        steps = list(range(0, 147, 3))  # 0, 3, 6, ..., 144
        for s in steps:
            label = f"T+{s:03d}h" if s > 0 else "T+000h (analisis)"
            self.step_combo.addItem(label, s)
        step_layout.addWidget(self.step_combo)
        layout.addLayout(step_layout)

        # Opcion: ultimo disponible
        self.latest_btn = QPushButton("Usar ultimo disponible")
        self.latest_btn.clicked.connect(self._use_latest)
        layout.addWidget(self.latest_btn)

        # Botones OK/Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._use_latest_flag = False

    def _preselect_nearest_run(self):
        """Pre-selecciona el run mas reciente disponible."""
        from datetime import datetime, timezone
        hour = datetime.now(timezone.utc).hour
        # Runs disponibles: 0, 6, 12, 18. Elegir el mas reciente ya pasado.
        valid_runs = [0, 6, 12, 18]
        # Encontrar el run mas reciente (con margen de ~5h para disponibilidad)
        best = 0
        for r in valid_runs:
            if r <= hour:
                best = r
        idx = valid_runs.index(best)
        self.run_combo.setCurrentIndex(idx)

    def _use_latest(self):
        self._use_latest_flag = True
        self.accept()

    def get_datetime(self) -> datetime | None:
        """Retorna la fecha seleccionada, o None si se eligio 'ultimo'."""
        if self._use_latest_flag:
            return None
        dt = self.datetime_edit.dateTime().toPyDateTime()
        # Ajustar hora al run seleccionado
        run_hour = int(self.run_combo.currentText()[:2])
        return dt.replace(hour=run_hour, minute=0, second=0, microsecond=0)

    def get_step(self) -> int:
        """Retorna el paso de prediccion seleccionado (0, 3, 6, ..., 144)."""
        return self.step_combo.currentData()


class ConfigDialog(QDialog):
    """Dialogo para ajustar parametros de configuracion."""

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Configuracion")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Grupo TFP
        tfp_group = QGroupBox("Parametros TFP")
        tfp_layout = QFormLayout()

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(1.0, 20.0)
        self.sigma_spin.setSingleStep(1.0)
        self.sigma_spin.setValue(cfg.tfp.smooth_sigma)
        tfp_layout.addRow("Suavizado sigma:", self.sigma_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(1e-6, 1e-4)
        self.threshold_spin.setDecimals(6)
        self.threshold_spin.setSingleStep(5e-6)
        self.threshold_spin.setValue(cfg.tfp.gradient_threshold)
        self.threshold_spin.setPrefix("  ")
        tfp_layout.addRow("Umbral gradiente (K/m):", self.threshold_spin)

        self.min_points_spin = QSpinBox()
        self.min_points_spin.setRange(3, 50)
        self.min_points_spin.setValue(cfg.tfp.min_front_points)
        tfp_layout.addRow("Min puntos frente:", self.min_points_spin)

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.1, 10.0)
        self.eps_spin.setSingleStep(0.1)
        self.eps_spin.setValue(cfg.tfp.dbscan_eps_deg)
        tfp_layout.addRow("DBSCAN eps (deg):", self.eps_spin)

        tfp_group.setLayout(tfp_layout)
        layout.addWidget(tfp_group)

        # Grupo Isobaras
        iso_group = QGroupBox("Isobaras")
        iso_layout = QFormLayout()

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 10)
        self.interval_spin.setValue(cfg.isobars.interval_hpa)
        iso_layout.addRow("Intervalo (hPa):", self.interval_spin)

        self.iso_sigma_spin = QDoubleSpinBox()
        self.iso_sigma_spin.setRange(0.0, 10.0)
        self.iso_sigma_spin.setSingleStep(0.5)
        self.iso_sigma_spin.setValue(cfg.isobars.smooth_sigma)
        iso_layout.addRow("Suavizado MSLP:", self.iso_sigma_spin)

        iso_group.setLayout(iso_layout)
        layout.addWidget(iso_group)

        # Botones
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply_to_config(self):
        """Aplica los valores del dialogo a la configuracion."""
        self.cfg.tfp.smooth_sigma = self.sigma_spin.value()
        self.cfg.tfp.gradient_threshold = self.threshold_spin.value()
        self.cfg.tfp.min_front_points = self.min_points_spin.value()
        self.cfg.tfp.dbscan_eps_deg = self.eps_spin.value()
        self.cfg.isobars.interval_hpa = self.interval_spin.value()
        self.cfg.isobars.smooth_sigma = self.iso_sigma_spin.value()


# Presets de campos para el mosaico
MOSAIC_PRESETS = {
    "Termodinamico": [
        "theta_e_850", "theta_e_700", "grad_theta_e_850",
        "grad_t_850", "thickness_1000_500", "temp_advection_850",
    ],
    "Cinematico": [
        "wind_speed_850", "wind_speed_500", "vorticity_850",
        "temp_advection_850", "grad_theta_e_850", "thickness_1000_500",
    ],
    "Completo": [
        "theta_e_850", "grad_theta_e_850", "thickness_1000_500",
        "temp_advection_850", "wind_speed_850", "vorticity_850",
    ],
}


class MosaicConfigDialog(QDialog):
    """Dialogo para configurar los paneles del mosaico."""

    def __init__(self, available_fields, current_selection=None, parent=None):
        """
        Args:
            available_fields: dict {clave: etiqueta} de campos disponibles.
            current_selection: lista de claves actualmente seleccionadas.
            parent: widget padre.
        """
        super().__init__(parent)
        self.setWindowTitle("Configurar mosaico")
        self.setMinimumWidth(500)

        self._available = available_fields
        # Excluir "none" del selector
        self._field_keys = [k for k in available_fields if k != "none"]
        self._field_labels = [available_fields[k] for k in self._field_keys]

        layout = QVBoxLayout(self)

        # --- Selector de layout ---
        layout_group = QGroupBox("Disposicion")
        layout_h = QHBoxLayout()
        self._layout_group = QButtonGroup(self)

        self._rb_2x2 = QRadioButton("2x2 (4 paneles)")
        self._rb_2x3 = QRadioButton("2x3 (6 paneles)")
        self._rb_3x3 = QRadioButton("3x3 (9 paneles)")
        self._layout_group.addButton(self._rb_2x2, 0)
        self._layout_group.addButton(self._rb_2x3, 1)
        self._layout_group.addButton(self._rb_3x3, 2)
        self._rb_2x3.setChecked(True)

        layout_h.addWidget(self._rb_2x2)
        layout_h.addWidget(self._rb_2x3)
        layout_h.addWidget(self._rb_3x3)
        layout_group.setLayout(layout_h)
        layout.addWidget(layout_group)

        self._layout_group.buttonClicked.connect(self._on_layout_changed)

        # --- Presets ---
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("Personalizado")
        for name in MOSAIC_PRESETS:
            self._preset_combo.addItem(name)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self._preset_combo)
        layout.addLayout(preset_layout)

        # --- Grid de combos ---
        self._combos_group = QGroupBox("Campos por panel")
        self._combos_layout = QGridLayout()
        self._combos_group.setLayout(self._combos_layout)
        layout.addWidget(self._combos_group)

        self._combos = []
        self._rebuild_combos(2, 3, current_selection)

        # --- Botones ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _rebuild_combos(self, nrows, ncols, selection=None):
        """Recrea los QComboBox para el layout dado."""
        # Limpiar combos anteriores
        for combo in self._combos:
            self._combos_layout.removeWidget(combo)
            combo.deleteLater()
        self._combos.clear()

        # Limpiar labels anteriores
        while self._combos_layout.count() > 0:
            item = self._combos_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        n_panels = nrows * ncols
        for i in range(n_panels):
            r = i // ncols
            c = i % ncols
            combo = QComboBox()
            for key, label in zip(self._field_keys, self._field_labels):
                combo.addItem(label, key)
            # Seleccionar campo por defecto
            if selection and i < len(selection):
                idx = self._field_keys.index(selection[i]) if selection[i] in self._field_keys else i % len(self._field_keys)
            else:
                idx = i % len(self._field_keys)
            combo.setCurrentIndex(idx)
            self._combos_layout.addWidget(QLabel(f"Panel {i+1}:"), r * 2, c)
            self._combos_layout.addWidget(combo, r * 2 + 1, c)
            self._combos.append(combo)

    def _on_layout_changed(self):
        """Cambia la grid de combos al cambiar layout."""
        nrows, ncols = self.get_layout()
        current = self.get_field_names()
        self._rebuild_combos(nrows, ncols, current)

    def _on_preset_changed(self, text):
        """Aplica un preset de campos."""
        if text in MOSAIC_PRESETS:
            fields = MOSAIC_PRESETS[text]
            for i, combo in enumerate(self._combos):
                if i < len(fields) and fields[i] in self._field_keys:
                    idx = self._field_keys.index(fields[i])
                    combo.setCurrentIndex(idx)

    def get_layout(self) -> tuple[int, int]:
        """Devuelve (nrows, ncols) seleccionado."""
        bid = self._layout_group.checkedId()
        if bid == 0:
            return 2, 2
        elif bid == 2:
            return 3, 3
        return 2, 3  # default

    def get_field_names(self) -> list[str]:
        """Devuelve lista de claves de campo seleccionadas."""
        return [combo.currentData() for combo in self._combos]
