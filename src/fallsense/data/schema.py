"""UP-Fall sensor-only CSV 的共同 schema 定義。"""

from __future__ import annotations

TIMESTAMP_COLUMN = "TimeStamps"
ID_COLUMNS = ("Subject", "Activity", "Trial")
TAG_COLUMN = "Tag"

IMU_POSITIONS = ("LeftAnkle", "RightPocket", "Waist", "Neck", "LeftWrist")
IMU_AXES = ("AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "Luminosity")
IMU_COLUMNS = tuple(f"{position}_{axis}" for position in IMU_POSITIONS for axis in IMU_AXES)
AMBIENT_COLUMNS = ("EEG", *(f"Infrared{i}" for i in range(1, 7)))
SENSOR_COLUMNS = (*IMU_COLUMNS, *AMBIENT_COLUMNS)
CSV_COLUMNS = (TIMESTAMP_COLUMN, *SENSOR_COLUMNS, *ID_COLUMNS, TAG_COLUMN)

EXPECTED_SENSOR_COUNT = 42

if len(SENSOR_COLUMNS) != EXPECTED_SENSOR_COUNT:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        f"預期 {EXPECTED_SENSOR_COUNT} 個 sensor channels，實際為 {len(SENSOR_COLUMNS)}"
    )
