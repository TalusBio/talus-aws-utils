"""Test cases for the s3 module."""
import json
from pathlib import Path
from unittest import TestCase

import pandas as pd
import pytest
from mypy_boto3_s3.service_resource import Bucket

from talus_aws_utils.s3 import _read_object
from talus_aws_utils.s3 import write_dataframe
from talus_aws_utils.s3 import write_json

DATA_DIR = Path(__file__).resolve().parent.joinpath("data")

PARQUET_FILE_KEY = "peptides_proteins_results.parquet"
CSV_FILE_KEY = "peptides_proteins_results.csv"
TSV_FILE_KEY = "subcellular_locations.tsv"
TXT_FILE_KEY = "proteins.txt"
JSON_FILE_KEY = "peptide_proteins.json"

PARQUET_EXPECTED = pd.read_parquet(DATA_DIR.joinpath(PARQUET_FILE_KEY))
CSV_EXPECTED = pd.read_csv(DATA_DIR.joinpath(CSV_FILE_KEY))
TSV_EXPECTED = pd.read_csv(DATA_DIR.joinpath(TSV_FILE_KEY), sep="\t")
TXT_EXPECTED = pd.read_csv(DATA_DIR.joinpath(TXT_FILE_KEY), sep="\t")
with open(DATA_DIR.joinpath(JSON_FILE_KEY), "r") as f:
    JSON_EXPECTED = json.load(f)


def test_write_dataframe_incorrect_format(bucket: Bucket) -> None:
    """Tests write_dataframe with an incorrect outputformat."""
    expected_error = (
        r"Invalid \(inferred\) outputformat. Use one of: parquet, txt, csv, tsv."
    )
    # outputformat given
    with pytest.raises(ValueError, match=expected_error):
        write_dataframe(
            dataframe=PARQUET_EXPECTED,
            bucket=bucket.name,
            key=PARQUET_FILE_KEY,
            outputformat=".elib",
        )


def test_write_dataframe_parquet(bucket: Bucket) -> None:
    """Tests write_dataframe for a parquet file."""
    # outputformat given
    write_dataframe(
        dataframe=PARQUET_EXPECTED,
        bucket=bucket.name,
        key=PARQUET_FILE_KEY,
        outputformat="parquet",
    )
    data_buffer = _read_object(bucket=bucket.name, key=PARQUET_FILE_KEY)
    parquet_actual = pd.read_parquet(data_buffer)
    pd.testing.assert_frame_equal(PARQUET_EXPECTED, parquet_actual)

    # outputformat inferred from filename
    write_dataframe(
        dataframe=PARQUET_EXPECTED, bucket=bucket.name, key=PARQUET_FILE_KEY
    )
    data_buffer = _read_object(bucket=bucket.name, key=PARQUET_FILE_KEY)
    parquet_actual = pd.read_parquet(data_buffer)
    pd.testing.assert_frame_equal(PARQUET_EXPECTED, parquet_actual)


def test_write_dataframe_csv(bucket: Bucket) -> None:
    """Tests write_dataframe for a csv file."""
    # outputformat given
    write_dataframe(
        dataframe=CSV_EXPECTED, bucket=bucket.name, key=CSV_FILE_KEY, outputformat="csv"
    )
    data_buffer = _read_object(bucket=bucket.name, key=CSV_FILE_KEY)
    csv_actual = pd.read_csv(data_buffer)
    pd.testing.assert_frame_equal(CSV_EXPECTED, csv_actual)

    # outputformat inferred from filename
    write_dataframe(dataframe=CSV_EXPECTED, bucket=bucket.name, key=CSV_FILE_KEY)
    data_buffer = _read_object(bucket=bucket.name, key=CSV_FILE_KEY)
    csv_actual = pd.read_csv(data_buffer)
    pd.testing.assert_frame_equal(CSV_EXPECTED, csv_actual)


def test_write_dataframe_tsv(bucket: Bucket) -> None:
    """Tests write_dataframe for a tsv file."""
    # outputformat given
    write_dataframe(
        dataframe=TSV_EXPECTED, bucket=bucket.name, key=TSV_FILE_KEY, outputformat="tsv"
    )
    data_buffer = _read_object(bucket=bucket.name, key=TSV_FILE_KEY)
    tsv_actual = pd.read_csv(data_buffer, sep="\t")
    pd.testing.assert_frame_equal(TSV_EXPECTED, tsv_actual)

    # outputformat inferred from filename
    write_dataframe(dataframe=TSV_EXPECTED, bucket=bucket.name, key=TSV_FILE_KEY)
    data_buffer = _read_object(bucket=bucket.name, key=TSV_FILE_KEY)
    tsv_actual = pd.read_csv(data_buffer, sep="\t")
    pd.testing.assert_frame_equal(TSV_EXPECTED, tsv_actual)


def test_write_dataframe_txt(bucket: Bucket) -> None:
    """Tests write_dataframe for a txt file."""
    # outputformat given
    write_dataframe(
        dataframe=TXT_EXPECTED, bucket=bucket.name, key=TXT_FILE_KEY, outputformat="txt"
    )
    data_buffer = _read_object(bucket=bucket.name, key=TXT_FILE_KEY)
    txt_actual = pd.read_csv(data_buffer, sep="\t")
    pd.testing.assert_frame_equal(TXT_EXPECTED, txt_actual)

    # outputformat inferred from filename
    write_dataframe(dataframe=TXT_EXPECTED, bucket=bucket.name, key=TXT_FILE_KEY)
    data_buffer = _read_object(bucket=bucket.name, key=TXT_FILE_KEY)
    txt_actual = pd.read_csv(data_buffer, sep="\t")
    pd.testing.assert_frame_equal(TXT_EXPECTED, txt_actual)


def test_write_json(bucket: Bucket) -> None:
    """Tests write_json."""
    write_json(dict_obj=JSON_EXPECTED, bucket=bucket.name, key=JSON_FILE_KEY)
    data_buffer = _read_object(bucket=bucket.name, key=JSON_FILE_KEY)
    json_actual = json.loads(data_buffer.read())
    TestCase().assertDictEqual(JSON_EXPECTED, json_actual)
