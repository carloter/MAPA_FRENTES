"""Tests para fronts/models.py."""

import numpy as np
import pytest

from mapa_frentes.fronts.models import Front, FrontCollection, FrontType


class TestFront:
    def test_creation(self, sample_front):
        assert sample_front.front_type == FrontType.COLD
        assert sample_front.npoints == 5
        assert sample_front.id == "test_cold_01"

    def test_coords_property(self, sample_front):
        coords = sample_front.coords
        assert coords.shape == (5, 2)
        # coords es [lon, lat]
        assert coords[0, 0] == -10.0  # lon
        assert coords[0, 1] == 40.0   # lat

    def test_set_coords(self, sample_front):
        new_coords = np.array([
            [-20.0, 50.0],
            [-19.0, 51.0],
        ])
        sample_front.set_coords(new_coords)
        assert sample_front.npoints == 2
        assert sample_front.lons[0] == -20.0
        assert sample_front.lats[0] == 50.0

    def test_auto_id(self):
        f = Front(FrontType.WARM, [0, 1], [0, 1])
        assert f.id.startswith("front_")

    def test_arrays_converted(self):
        f = Front(FrontType.COLD, [1.0, 2.0], [3.0, 4.0])
        assert isinstance(f.lats, np.ndarray)
        assert isinstance(f.lons, np.ndarray)


class TestFrontCollection:
    def test_add_remove(self, sample_collection):
        assert len(sample_collection) == 2
        sample_collection.remove("test_cold_01")
        assert len(sample_collection) == 1

    def test_get_by_id(self, sample_collection):
        f = sample_collection.get_by_id("test_warm_01")
        assert f is not None
        assert f.front_type == FrontType.WARM

    def test_get_nonexistent(self, sample_collection):
        assert sample_collection.get_by_id("nonexistent") is None

    def test_clear(self, sample_collection):
        sample_collection.clear()
        assert len(sample_collection) == 0

    def test_iteration(self, sample_collection):
        types = [f.front_type for f in sample_collection]
        assert FrontType.COLD in types
        assert FrontType.WARM in types

    def test_metadata(self, sample_collection):
        assert sample_collection.valid_time == "2025-05-01T00:00"
        assert sample_collection.model_run == "2025050100"
