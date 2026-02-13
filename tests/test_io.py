"""Tests para fronts/io.py - serializacion GeoJSON."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mapa_frentes.fronts.io import (
    collection_to_geojson,
    collection_from_geojson,
    save_session,
    load_session,
)
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType


class TestGeoJSONSerialization:
    def test_roundtrip(self, sample_collection):
        """Serializar y deserializar debe conservar los datos."""
        geojson_data = collection_to_geojson(sample_collection)
        restored = collection_from_geojson(geojson_data)

        assert len(restored) == len(sample_collection)
        assert restored.valid_time == sample_collection.valid_time
        assert restored.model_run == sample_collection.model_run

    def test_front_types_preserved(self, sample_collection):
        geojson_data = collection_to_geojson(sample_collection)
        restored = collection_from_geojson(geojson_data)

        types = {f.id: f.front_type for f in restored}
        assert types["test_cold_01"] == FrontType.COLD
        assert types["test_warm_01"] == FrontType.WARM

    def test_coordinates_preserved(self, sample_collection):
        geojson_data = collection_to_geojson(sample_collection)
        restored = collection_from_geojson(geojson_data)

        original = sample_collection.get_by_id("test_cold_01")
        restored_front = restored.get_by_id("test_cold_01")

        np.testing.assert_array_almost_equal(
            original.lats, restored_front.lats
        )
        np.testing.assert_array_almost_equal(
            original.lons, restored_front.lons
        )

    def test_geojson_structure(self, sample_collection):
        data = collection_to_geojson(sample_collection)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 2
        assert data["features"][0]["type"] == "Feature"
        assert data["features"][0]["geometry"]["type"] == "LineString"

    def test_empty_collection(self):
        empty = FrontCollection()
        data = collection_to_geojson(empty)
        restored = collection_from_geojson(data)
        assert len(restored) == 0


class TestFileIO:
    def test_save_load_roundtrip(self, sample_collection):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_session.geojson"

            save_session(sample_collection, filepath)
            assert filepath.exists()

            loaded = load_session(filepath)
            assert len(loaded) == len(sample_collection)

            # Verificar contenido JSON valido
            with open(filepath) as f:
                data = json.load(f)
            assert data["type"] == "FeatureCollection"

    def test_save_creates_parent_dirs(self, sample_collection):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "deep" / "session.geojson"

            save_session(sample_collection, filepath)
            assert filepath.exists()
