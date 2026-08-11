from fallsense.data.schema import (
    CSV_COLUMNS,
    ID_COLUMNS,
    SENSOR_COLUMNS,
    TAG_COLUMN,
    TIMESTAMP_COLUMN,
)


def test_sensor_schema_has_expected_shape() -> None:
    assert len(SENSOR_COLUMNS) == 42
    assert len(CSV_COLUMNS) == 47
    assert CSV_COLUMNS[0] == TIMESTAMP_COLUMN
    assert CSV_COLUMNS[-4:-1] == ID_COLUMNS
    assert CSV_COLUMNS[-1] == TAG_COLUMN
