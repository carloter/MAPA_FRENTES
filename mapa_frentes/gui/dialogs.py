"""Dialogos de la aplicacion: selector de fecha, configuracion."""

from datetime import datetime

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QDateTimeEdit, QComboBox, QPushButton, QDialogButtonBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QFileDialog, QMessageBox,
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
