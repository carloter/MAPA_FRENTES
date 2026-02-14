"""Serializacion de frentes a/desde GeoJSON."""

import json
import logging
from pathlib import Path

import geojson
import numpy as np

from mapa_frentes.fronts.models import (
    Front,
    FrontCollection,
    FrontType,
    CycloneSystem,
    CycloneSystemCollection,
)
from mapa_frentes.analysis.pressure_centers import PressureCenter

logger = logging.getLogger(__name__)


def collection_to_geojson(collection: FrontCollection) -> dict:
    """Convierte FrontCollection a un dict GeoJSON FeatureCollection.

    Cada frente se serializa como un Feature con geometria LineString
    y propiedades (tipo, id).
    """
    features = []
    for front in collection:
        coords = list(zip(
            front.lons.tolist(),
            front.lats.tolist(),
        ))
        feature = geojson.Feature(
            geometry=geojson.LineString(coords),
            properties={
                "id": front.id,
                "front_type": front.front_type.value,
                "flip_symbols": front.flip_symbols,
                "center_id": front.center_id,
                "association_end": front.association_end,
                "occlusion_score": front.occlusion_score,
                "occlusion_type": front.occlusion_type,
                "importance_score": front.importance_score,
                "is_primary": front.is_primary,
            },
        )
        features.append(feature)

    fc = geojson.FeatureCollection(
        features,
        properties={
            "valid_time": collection.valid_time,
            "model_run": collection.model_run,
            "description": collection.description,
        },
    )
    return fc


def collection_from_geojson(data: dict) -> FrontCollection:
    """Crea FrontCollection desde un dict GeoJSON FeatureCollection."""
    props = data.get("properties", {})
    collection = FrontCollection(
        valid_time=props.get("valid_time", ""),
        model_run=props.get("model_run", ""),
        description=props.get("description", ""),
    )

    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        fprops = feature.get("properties", {})

        if geom.get("type") != "LineString":
            continue

        coords = np.array(geom["coordinates"])
        lons = coords[:, 0]
        lats = coords[:, 1]

        front_type_str = fprops.get("front_type", "cold")
        try:
            front_type = FrontType(front_type_str)
        except ValueError:
            front_type = FrontType.COLD

        front = Front(
            front_type=front_type,
            lats=lats,
            lons=lons,
            id=fprops.get("id", ""),
            flip_symbols=fprops.get("flip_symbols", False),
            center_id=fprops.get("center_id", ""),
            association_end=fprops.get("association_end", ""),
            occlusion_score=fprops.get("occlusion_score", 0.0),
            occlusion_type=fprops.get("occlusion_type", ""),
            importance_score=fprops.get("importance_score", 0.0),
            is_primary=fprops.get("is_primary", False),
        )
        collection.add(front)

    return collection


def save_session(collection: FrontCollection, filepath: str | Path):
    """Guarda FrontCollection como fichero GeoJSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = collection_to_geojson(collection)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Sesion guardada: %s (%d frentes)", filepath, len(collection))


def load_session(filepath: str | Path) -> FrontCollection:
    """Carga FrontCollection desde fichero GeoJSON."""
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    collection = collection_from_geojson(data)
    logger.info("Sesion cargada: %s (%d frentes)", filepath, len(collection))
    return collection


def systems_to_geojson(systems: CycloneSystemCollection) -> dict:
    """Convierte CycloneSystemCollection a GeoJSON FeatureCollection.

    Cada sistema se serializa como un Feature con:
    - Geometría: MultiLineString con todos los frentes del sistema
    - Properties: info del centro, nombre, metadata, lista de frentes
    """
    features = []

    for system in systems:
        # Geometría: MultiLineString con todos los frentes
        lines = []
        front_ids = []
        for front in system.fronts:
            coords = list(zip(front.lons.tolist(), front.lats.tolist()))
            lines.append(coords)
            front_ids.append(front.id)

        if not lines:
            # Sistema sin frentes: usar Point del centro
            geometry = geojson.Point([system.center.lon, system.center.lat])
        else:
            geometry = geojson.MultiLineString(lines)

        # Properties del sistema
        properties = {
            "system_id": system.id,
            "center_lat": system.center.lat,
            "center_lon": system.center.lon,
            "center_pressure": system.center.value,
            "center_type": system.center.type,
            "is_primary": system.is_primary,
            "name": system.name,
            "valid_time": system.valid_time,
            "metadata": system.metadata,
            "front_ids": front_ids,
            "n_fronts": len(system.fronts),
            "n_secondary_centers": len(system.secondary_centers),
        }

        feature = geojson.Feature(geometry=geometry, properties=properties)
        features.append(feature)

    # Añadir frentes no asociados como Features individuales
    for front in systems.unassociated_fronts:
        coords = list(zip(front.lons.tolist(), front.lats.tolist()))
        feature = geojson.Feature(
            geometry=geojson.LineString(coords),
            properties={
                "id": front.id,
                "front_type": front.front_type.value,
                "unassociated": True,
            },
        )
        features.append(feature)

    fc = geojson.FeatureCollection(
        features,
        properties={
            "valid_time": systems.valid_time,
            "n_systems": len(systems),
            "n_unassociated_fronts": len(systems.unassociated_fronts),
        },
    )
    return fc


def save_systems(systems: CycloneSystemCollection, filepath: str | Path):
    """Guarda CycloneSystemCollection como fichero GeoJSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = systems_to_geojson(systems)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(
        "Sistemas guardados: %s (%d sistemas, %d frentes no asociados)",
        filepath, len(systems), len(systems.unassociated_fronts)
    )
