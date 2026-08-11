from __future__ import annotations

import csv
from pathlib import Path

from fallsense.data.io import estimated_sample_rate_hz, read_trial_csv
from fallsense.data.schema import CSV_COLUMNS, SENSOR_COLUMNS


def test_read_official_uneven_two_row_header_with_hidden_tag(tmp_path: Path) -> None:
    path = tmp_path / "Subject1Activity1Trial1.csv"
    header = list(CSV_COLUMNS[:-1])
    subheader = [""] * 43
    values = ["2018-01-01T00:00:00.000000", *(["0.5"] * len(SENSOR_COLUMNS)), "1", "1", "1", "7"]
    second_values = [
        "2018-01-01T00:00:00.050000",
        *(["0.6"] * len(SENSOR_COLUMNS)),
        "1",
        "1",
        "1",
        "1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows((header, subheader, values, second_values))

    frame = read_trial_csv(path)
    assert tuple(frame.columns) == CSV_COLUMNS
    assert frame["Tag"].tolist() == [7, 1]
    assert frame["TimeStamps"].tolist() == [0.0, 0.05]
    assert estimated_sample_rate_hz(frame) == 20.0
