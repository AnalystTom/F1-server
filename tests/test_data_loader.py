from pathlib import Path
import pytest

from src.data_loader import CSVContextLoader


def test_load_as_text_success(gcs_test_bucket):
    loader = CSVContextLoader(gcs_test_bucket)
    text = loader.load_as_text(["drivers_F1.csv", "lap_times_F1.csv"])
    assert "drivers_F1.csv" in text
    assert "lap_times_F1.csv" in text
    assert "P GASLY" in text


def test_load_as_text_missing(gcs_test_bucket):
    loader = CSVContextLoader(gcs_test_bucket)
    with pytest.raises(FileNotFoundError):
        loader.load_as_text(["not_exists.csv"])
