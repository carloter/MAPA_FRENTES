"""Modelos de datos para frentes meteorologicos."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mapa_frentes.analysis.pressure_centers import PressureCenter


class FrontType(Enum):
    """Tipos de frente meteorologico segun WMO."""
    COLD = "cold"
    WARM = "warm"
    OCCLUDED = "occluded"  # mantener para compatibilidad
    COLD_OCCLUDED = "cold_occluded"  # oclusión de tipo frío
    WARM_OCCLUDED = "warm_occluded"  # oclusión de tipo cálido
    WARM_SECLUSION = "warm_seclusion"  # warm-core seclusion
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
        flip_symbols: Si invertir simbolos del frente.
        center_id: ID del centro de presion asociado.
        association_end: "start" o "end" indica extremo asociado al centro.
        occlusion_score: Confianza de deteccion de oclusion (0-1).
        occlusion_type: Subtipo especifico de oclusion.
        importance_score: Score de importancia del frente (0-1).
        is_primary: True si es frente principal, False si es secundario.
    """
    front_type: FrontType
    lats: np.ndarray
    lons: np.ndarray
    id: str = ""
    flip_symbols: bool = False
    center_id: str = ""
    association_end: str = ""
    occlusion_score: float = 0.0
    occlusion_type: str = ""
    importance_score: float = 0.0
    is_primary: bool = False

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
    metadata: dict = field(default_factory=dict)

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


@dataclass
class CycloneSystem:
    """Sistema ciclónico: centro de baja + frentes asociados."""
    center: "PressureCenter"  # Centro L primario
    fronts: List[Front] = field(default_factory=list)
    secondary_centers: List["PressureCenter"] = field(default_factory=list)
    name: str = ""  # Nombre borrasca
    valid_time: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """ID del sistema (usa el ID del centro primario)."""
        return self.center.id

    @property
    def is_primary(self) -> bool:
        """True si el centro primario del sistema es primario."""
        return self.center.primary

    def get_fronts_by_type(self, ftype: FrontType) -> List[Front]:
        """Obtiene todos los frentes de un tipo especifico."""
        return [f for f in self.fronts if f.front_type == ftype]

    def compute_statistics(self):
        """Calcula metadata: n_fronts, extension, longitud total, etc."""
        if not self.fronts:
            self.metadata = {
                "n_fronts": 0,
                "n_cold": 0,
                "n_warm": 0,
                "n_occluded": 0,
                "total_length_deg": 0.0,
            }
            return

        # Contar frentes por tipo
        n_cold = sum(1 for f in self.fronts if f.front_type == FrontType.COLD)
        n_warm = sum(1 for f in self.fronts if f.front_type == FrontType.WARM)
        n_occluded = sum(1 for f in self.fronts if f.front_type in (
            FrontType.OCCLUDED, FrontType.COLD_OCCLUDED,
            FrontType.WARM_OCCLUDED, FrontType.WARM_SECLUSION
        ))

        # Calcular longitud total
        total_length = 0.0
        for front in self.fronts:
            coords = front.coords
            if len(coords) > 1:
                diffs = np.diff(coords, axis=0)
                seg_lengths = np.sqrt((diffs**2).sum(axis=1))
                total_length += seg_lengths.sum()

        # Bounding box
        all_lats = np.concatenate([f.lats for f in self.fronts])
        all_lons = np.concatenate([f.lons for f in self.fronts])

        self.metadata = {
            "n_fronts": len(self.fronts),
            "n_cold": n_cold,
            "n_warm": n_warm,
            "n_occluded": n_occluded,
            "total_length_deg": float(total_length),
            "lat_min": float(all_lats.min()),
            "lat_max": float(all_lats.max()),
            "lon_min": float(all_lons.min()),
            "lon_max": float(all_lons.max()),
        }


@dataclass
class CycloneSystemCollection:
    """Coleccion de sistemas ciclonicos."""
    systems: List[CycloneSystem] = field(default_factory=list)
    unassociated_fronts: List[Front] = field(default_factory=list)
    valid_time: str = ""

    def get_by_id(self, system_id: str) -> CycloneSystem | None:
        """Busca un sistema por su ID."""
        for system in self.systems:
            if system.id == system_id:
                return system
        return None

    def __len__(self):
        return len(self.systems)

    def __iter__(self):
        return iter(self.systems)
