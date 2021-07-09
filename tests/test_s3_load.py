"""Test cases for the s3 module."""
import json
import os

from pathlib import Path
from typing import Iterable
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from mypy_boto3_s3.service_resource import Bucket

from talus_aws_utils.s3 import (
    file_exists_in_bucket,
    file_keys_in_bucket,
    file_size,
    read_dataframe,
    read_json,
    read_numpy_array,
)


DATA_DIR = Path(__file__).resolve().parent.joinpath("data")

PARQUET_FILE_KEY = "peptides_proteins_results.parquet"
CSV_FILE_KEY = "peptides_proteins_results.csv"
TSV_FILE_KEY = "subcellular_locations.tsv"
TXT_FILE_KEY = "proteins.txt"
JSON_FILE_KEY = "peptide_proteins.json"
NP_ARRAY_FILE_KEY = "zeros_array.npy"

PARQUET_EXPECTED = pd.read_parquet(DATA_DIR.joinpath(PARQUET_FILE_KEY))
CSV_EXPECTED = pd.read_csv(DATA_DIR.joinpath(CSV_FILE_KEY))
TSV_EXPECTED = pd.read_csv(DATA_DIR.joinpath(TSV_FILE_KEY), sep="\t")
TXT_EXPECTED = pd.read_csv(DATA_DIR.joinpath(TXT_FILE_KEY), sep="\t")
NP_ARRAY_EXPECTED = np.load(DATA_DIR.joinpath(NP_ARRAY_FILE_KEY))
with open(DATA_DIR.joinpath(JSON_FILE_KEY), "r") as f:
    JSON_EXPECTED = json.load(f)


@pytest.fixture
def loaded_bucket(bucket: Bucket) -> Iterable[Bucket]:
    """Fixture for a bucket with uploaded files."""
    bucket.upload_file(
        Filename=os.path.join(DATA_DIR, PARQUET_FILE_KEY),
        Key=PARQUET_FILE_KEY,
    )
    bucket.upload_file(
        Filename=os.path.join(DATA_DIR, CSV_FILE_KEY),
        Key=CSV_FILE_KEY,
    )
    bucket.upload_file(
        Filename=os.path.join(DATA_DIR, TSV_FILE_KEY),
        Key=TSV_FILE_KEY,
    )
    bucket.upload_file(
        Filename=os.path.join(DATA_DIR, TXT_FILE_KEY),
        Key=TXT_FILE_KEY,
    )
    bucket.upload_file(
        Filename=os.path.join(DATA_DIR, TXT_FILE_KEY),
        Key=TXT_FILE_KEY,
    )
    bucket.upload_file(
        Filename=os.path.join(DATA_DIR, JSON_FILE_KEY),
        Key=JSON_FILE_KEY,
    )
    bucket.upload_file(
        Filename=os.path.join(DATA_DIR, NP_ARRAY_FILE_KEY),
        Key=NP_ARRAY_FILE_KEY,
    )
    yield bucket


def test_read_dataframe_incorrect_format(loaded_bucket: Bucket) -> None:
    """Tests read_dataframe with an incorrect inputformat."""
    expected_error = (
        r"Invalid \(inferred\) inputformat. Use one of: parquet, txt, csv, tsv."
    )
    # inputformat given
    with pytest.raises(ValueError, match=expected_error):
        _ = read_dataframe(
            bucket=loaded_bucket.name, key=PARQUET_FILE_KEY, inputformat=".elib"
        )


def test_read_dataframe_file_doesnt_exist(loaded_bucket: Bucket) -> None:
    """Tests read_dataframe with a nonexisting file."""
    expected_error = r"File doesn't exist."
    with pytest.raises(ValueError, match=expected_error):
        _ = read_dataframe(bucket=loaded_bucket.name, key="random_file.elib")


def test_read_dataframe_parquet(loaded_bucket: Bucket) -> None:
    """Tests read_dataframe for a parquet file."""
    # inputformat given
    parquet_actual = read_dataframe(
        bucket=loaded_bucket.name, key=PARQUET_FILE_KEY, inputformat="parquet"
    )
    pd.testing.assert_frame_equal(PARQUET_EXPECTED, parquet_actual)

    # inputformat inferred from filename
    parquet_actual = read_dataframe(bucket=loaded_bucket.name, key=PARQUET_FILE_KEY)
    pd.testing.assert_frame_equal(PARQUET_EXPECTED, parquet_actual)


def test_read_dataframe_csv(loaded_bucket: Bucket) -> None:
    """Tests read_dataframe for a csv file."""
    # inputformat given
    csv_actual = read_dataframe(
        bucket=loaded_bucket.name, key=CSV_FILE_KEY, inputformat="csv"
    )
    pd.testing.assert_frame_equal(CSV_EXPECTED, csv_actual)

    # inputformat inferred from filename
    csv_actual = read_dataframe(bucket=loaded_bucket.name, key=CSV_FILE_KEY)
    pd.testing.assert_frame_equal(CSV_EXPECTED, csv_actual)


def test_read_dataframe_tsv(loaded_bucket: Bucket) -> None:
    """Tests read_dataframe for a tsv file."""
    # inputformat given
    tsv_actual = read_dataframe(
        bucket=loaded_bucket.name, key=TSV_FILE_KEY, inputformat="tsv"
    )
    pd.testing.assert_frame_equal(TSV_EXPECTED, tsv_actual)

    # inputformat inferred from filename
    tsv_actual = read_dataframe(bucket=loaded_bucket.name, key=TSV_FILE_KEY)
    pd.testing.assert_frame_equal(TSV_EXPECTED, tsv_actual)


def test_read_dataframe_txt(loaded_bucket: Bucket) -> None:
    """Tests read_dataframe for a txt file."""
    # inputformat given
    txt_actual = read_dataframe(
        bucket=loaded_bucket.name, key=TXT_FILE_KEY, inputformat="txt"
    )
    pd.testing.assert_frame_equal(TXT_EXPECTED, txt_actual)

    # inputformat inferred from filename
    txt_actual = read_dataframe(bucket=loaded_bucket.name, key=TXT_FILE_KEY)
    pd.testing.assert_frame_equal(TXT_EXPECTED, txt_actual)


def test_read_json(loaded_bucket: Bucket) -> None:
    """Tests read_json."""
    json_actual = read_json(bucket=loaded_bucket.name, key=JSON_FILE_KEY)
    TestCase().assertDictEqual(JSON_EXPECTED, json_actual)


def test_read_numpy_array(loaded_bucket: Bucket) -> None:
    """Tests read_numpy_array."""
    np_array_actual = read_numpy_array(bucket=loaded_bucket.name, key=NP_ARRAY_FILE_KEY)
    np.testing.assert_equal(np_array_actual, NP_ARRAY_EXPECTED)


def test_file_keys_in_bucket(loaded_bucket: Bucket) -> None:
    """Tests file_keys_in_bucket."""
    file_keys_expected = [
        CSV_FILE_KEY,
        TSV_FILE_KEY,
        PARQUET_FILE_KEY,
        JSON_FILE_KEY,
        NP_ARRAY_FILE_KEY,
        TXT_FILE_KEY,
    ]
    file_keys_actual = file_keys_in_bucket(bucket=loaded_bucket.name, key="")

    assert set(file_keys_expected) == set(file_keys_actual)


def test_file_keys_in_bucket_csv_only(loaded_bucket: Bucket) -> None:
    """Tests file_keys_in_bucket using the file_type filter."""
    file_keys_expected = [CSV_FILE_KEY]
    file_keys_actual = file_keys_in_bucket(
        bucket=loaded_bucket.name, key="", file_type="csv"
    )

    assert set(file_keys_expected) == set(file_keys_actual)


def test_file_exists_in_bucket(loaded_bucket: Bucket) -> None:
    """Tests file_exists_in_bucket."""
    assert file_exists_in_bucket(bucket=loaded_bucket.name, key=CSV_FILE_KEY)
    assert not file_exists_in_bucket(bucket=loaded_bucket.name, key="random_file.csv")


def test_file_size(loaded_bucket: Bucket) -> None:
    """Tests file_size."""
    size_expected = set(["627B", "628B", "629B"])
    raw_size_expected = set(["627", "628", "629"])
    size_actual = file_size(bucket=loaded_bucket.name, key=CSV_FILE_KEY)
    assert size_actual in size_expected
    raw_size_actual = file_size(
        bucket=loaded_bucket.name, key=CSV_FILE_KEY, raw_size=True
    )
    assert raw_size_actual in raw_size_expected


def test_file_size_nonexistent(loaded_bucket: Bucket) -> None:
    """Tests file_size where file doesn't exist."""
    expected_error = "File doesn't exist. Couldn't retrieve file size."
    with pytest.raises(ValueError, match=expected_error):
        _ = file_size(bucket=loaded_bucket.name, key="random_file.csv")
