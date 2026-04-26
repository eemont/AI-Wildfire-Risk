import duckdb
import pytest
from fastapi.testclient import TestClient

from ai_wildfire_tracker.api import server

FIRE_ROW = (34.0, -118.0, 350.0, 300.0, 50.0, "high", "2024-01-01", "1200")


@pytest.fixture(autouse=True)
def reset_model_state(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("ALLOW_MODEL_FALLBACK", raising=False)
    server._model = None
    server._mark_model_status("not_loaded")
    yield
    server._model = None
    server._mark_model_status("not_loaded")


def test_missing_default_model_uses_reported_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MODEL_PATH", tmp_path / "missing-model.joblib")

    scores = server.compute_risk_batch([FIRE_ROW], weather_map={}, env_map={})

    assert scores == [server._fallback_risk(350.0, 50.0)]
    health = server.health()
    assert health["status"] == "degraded"
    assert health["model_loaded"] is False
    assert health["model_status"] == "fallback"
    assert health["fallback_enabled"] is True


def test_explicit_missing_model_path_disables_fallback(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing-model.joblib"
    monkeypatch.setenv("MODEL_PATH", str(missing_model))
    monkeypatch.setattr(server, "MODEL_PATH", missing_model)

    with pytest.raises(server.ModelUnavailableError):
        server._load_model()

    health = server.health()
    assert health["status"] == "error"
    assert health["model_loaded"] is False
    assert health["model_status"] == "unavailable"
    assert health["fallback_enabled"] is False


def test_fires_returns_503_when_required_model_is_missing(monkeypatch, tmp_path):
    db_path = tmp_path / "fires.db"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE fires (
            latitude DOUBLE,
            longitude DOUBLE,
            bright_ti4 DOUBLE,
            bright_ti5 DOUBLE,
            frp DOUBLE,
            confidence VARCHAR,
            acq_date VARCHAR,
            acq_time VARCHAR
        )
        """
    )
    con.execute("INSERT INTO fires VALUES (?, ?, ?, ?, ?, ?, ?, ?)", FIRE_ROW)
    con.close()

    missing_model = tmp_path / "missing-model.joblib"
    monkeypatch.setenv("MODEL_PATH", str(missing_model))
    monkeypatch.setattr(server, "MODEL_PATH", missing_model)
    monkeypatch.setattr(server, "DB_PATH", str(db_path))

    response = TestClient(server.app).get("/fires")

    assert response.status_code == 503
    assert response.json()["detail"] == "RF model is unavailable and fallback is disabled"
