"""Serializacion de frentes a/desde GeoJSON."""

import json
import logging
from pathlib import Path

import geojson
import numpy as np

from mapa_frentes.fronts.models import Front, FrontCollection, FrontType

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
