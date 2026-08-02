from __future__ import annotations

import zipfile
from pathlib import Path

from city2graph.transportation import load_gtfs

_STOP_TIMES = (
    "trip_id,arrival_time,departure_time,stop_id,stop_sequence,"
    "stop_headsign,pickup_type,drop_off_time,shape_dist_traveled\n"
    "T1,6:00:00,6:00:00,S1,1\n"
    "T1,6:10:00,6:10:00,S2,2\n"
    "T2,7:00:00,7:00:00,S1,1,Airport,0,7:00:00,0\n"
    "T2,7:15:00,7:15:00,S2,2,Airport,0,7:15:00,1200\n"
)
_STOPS = "stop_id,stop_name,stop_lat,stop_lon\nS1,First,36.9,-116.7\nS2,Second,36.9,-116.8\n"


def test_load_gtfs_parses_rows_with_omitted_optional_fields(tmp_path: Path) -> None:
    feed = tmp_path / "ragged.zip"
    with zipfile.ZipFile(feed, "w") as z:
        z.writestr("stop_times.txt", _STOP_TIMES)
        z.writestr("stops.txt", _STOPS)

    con = load_gtfs(str(feed))
    cols = [r[0] for r in con.execute("DESCRIBE stop_times").fetchall()]

    assert len(cols) == 9, f"stop_times mis-parsed into {len(cols)} column(s): {cols[0][:60]!r}"
    assert "stop_sequence" in cols
    assert con.execute("SELECT count(*) FROM stop_times").fetchone()[0] == 4
