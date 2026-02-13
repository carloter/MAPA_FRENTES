"""Modelos de datos para frentes meteorologicos."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np


class FrontType(Enum):
    """Tipos de frente meteorologico segun WMO."""
    COLD = "cold"
    WARM = "warm"
    OCCLUDED = "occluded"
    STATIONARY = "stationary"
    INSTABILITY_LINE = "instability_line"


@dataclass
class Front:
    """Representa un frente meteorologico como polilínea.

    Attributes:
        front_type: Tipo de frente (frio, calido, ocluido, estacionario).
        lats: Latitudes de los vertices.
        lons: Longitudes de los vertices.
        id: Identificador unico (generado automaticamente).
    """
    front_type: FrontType
    lats: np.ndarray
    lons: np.ndarray
    id: str = ""
    flip_symbols: bool = False
    center_id: str = ""

    def __post_init__(self):
        self.lats = np.asarray(self.lats, dtype=float)
        self.lons = np.asarray(self.lons, dtype=float)
        if not self.id:
            self.id = f"front_{id(self)}"

    @property
    def npoints(self) -> int:
        return len(self.lats)

    @property
    def coords(self) -> np.ndarray:
        """Array (N, 2) con columnas [lon, lat]."""
        return np.column_stack([self.lons, self.lats])

    def set_coords(self, coords: np.ndarray):
        """Actualiza coordenadas desde array (N, 2) [lon, lat]."""
        self.lons = coords[:, 0].copy()
        self.lats = coords[:, 1].copy()


@dataclass
class FrontCollection:
    """Coleccion de frentes con metadatos de la sesion."""
    fronts: List[Front] = field(default_factory=list)
    valid_time: str = ""
    model_run: str = ""
    description: str = ""

    def add(self, front: Front):
        self.fronts.append(front)

    def remove(self, front_id: str):
        self.fronts = [f for f in self.fronts if f.id != front_id]

    def get_by_id(self, front_id: str) -> Front | None:
        for f in self.fronts:
            if f.id == front_id:
                return f
        return None

    def clear(self):
        self.fronts.clear()

    def __len__(self):
        return len(self.fronts)

    def __iter__(self):
        return iter(self.fronts)
