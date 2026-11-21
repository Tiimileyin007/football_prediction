from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from footmodel.value import collector, store
from footmodel.value.sources import Quote

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _run(db):
    import os
    os.environ["FOOTMODEL_DB"] = str(db)
    at = st_testing.AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_starts_on_an_empty_database(tmp_path):
    at = _run(tmp_path / "e.db")
    assert not at.exception
    assert any("No upcoming fixtures" in i.value for i in at.info)


def test_app_shows_fixtures_and_value(tmp_path):
    db = tmp_path / "s.db"
    con = store.connect(str(db))
    k = "2099-01-01T15:00:00"
    collector.snapshot(con, [Quote(k, "Arsenal", "Chelsea", "pinnacle", "1x2", s, o) for s, o in zip("HDA", (2.0, 3.5, 3.9))], "2098-12-01T00:00:00")
    for s, o in zip("HDA", (2.4, 3.4, 3.8)):
        store.add_quote(con, store.match_key(k, "Arsenal", "Chelsea"), "sportybet", "1x2", s, o, "2098-12-02T00:00:00")
    con.commit()
    con.close()
    at = _run(db)
    assert not at.exception
    assert len(at.dataframe) >= 1
