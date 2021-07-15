"""Test cases for the s3 module."""
import json
import pickle

from pathlib import Path
from unittest import TestCase

import joblib
import numpy as np
import pandas as pd
import pytest

from mypy_boto3_s3.service_resource import Bucket

import talus_aws_utils.s3 as s3_utils


DATA_DIR = Path(__file__).resolve().parent.joinpath("data")

PARQUET_FILE_KEY = "peptides_proteins_results.parquet"
CSV_FILE_KEY = "peptides_proteins_results.csv"
TSV_FILE_KEY = "subcellular_locations.tsv"
TXT_FILE_KEY = "proteins.txt"
JSON_FILE_KEY = "peptide_proteins.json"
NP_ARRAY_FILE_KEY = "zeros_array.npy"
JOBLIB_FILE_KEY = "test.joblib"

PARQUET_EXPECTED = pd.read_parquet(DATA_DIR.joinpath(PARQUET_FILE_KEY))
CSV_EXPECTED = pd.read_csv(DATA_DIR.joinpath(CSV_FILE_KEY))
TSV_EXPECTED = pd.read_csv(DATA_DIR.joinpath(TSV_FILE_KEY), sep="\t")
TXT_EXPECTED = pd.read_csv(DATA_DIR.joinpath(TXT_FILE_KEY), sep="\t")
NP_ARRAY_EXPECTED = np.load(DATA_DIR.joinpath(NP_ARRAY_FILE_KEY))
JOBLIB_EXPECTED = joblib.load(DATA_DIR.joinpath(JOBLIB_FILE_KEY))
with open(DATA_DIR.joinpath(JSON_FILE_KEY), "r") as f:
    JSON_EXPECTED = json.load(f)


def test_write_dataframe_incorrect_format(bucket: Bucket) -> None:
    """Tests write_dataframe with an incorrect outputformat."""
    expected_error = (
        r"Invalid \(inferred\) outputformat. Use one of: parquet, txt, csv, tsv."
    )
    # outputformat given
    with pytest.raises(ValueError, match=expected_error):
        s3_utils.write_dataframe(
            dataframe=PARQUET_EXPECTED,
            bucket=bucket.name,
            key=PARQUET_FILE_KEY,
            outputformat=".elib",
        )


def test_write_dataframe_parquet(bucket: Bucket) -> None:
    """Tests write_dataframe for a parquet file."""
    # outputformat given
    s3_utils.write_dataframe(
        dataframe=PARQUET_EXPECTED,
        bucket=bucket.name,
        key=PARQUET_FILE_KEY,
        outputformat="parquet",
    )
    data_buffer = s3_utils._read_object(bucket=bucket.name, key=PARQUET_FILE_KEY)
    parquet_actual = pd.read_parquet(data_buffer)
    pd.testing.assert_frame_equal(PARQUET_EXPECTED, parquet_actual)

    # outputformat inferred from filename
    s3_utils.write_dataframe(
        dataframe=PARQUET_EXPECTED, bucket=bucket.name, key=PARQUET_FILE_KEY
    )
    data_buffer = s3_utils._read_object(bucket=bucket.name, key=PARQUET_FILE_KEY)
    parquet_actual = pd.read_parquet(data_buffer)
    pd.testing.assert_frame_equal(PARQUET_EXPECTED, parquet_actual)


def test_write_dataframe_csv(bucket: Bucket) -> None:
    """Tests write_dataframe for a csv file."""
    # outputformat given
    s3_utils.write_dataframe(
        dataframe=CSV_EXPECTED, bucket=bucket.name, key=CSV_FILE_KEY, outputformat="csv"
    )
    data_buffer = s3_utils._read_object(bucket=bucket.name, key=CSV_FILE_KEY)
    csv_actual = pd.read_csv(data_buffer)
    pd.testing.assert_frame_equal(CSV_EXPECTED, csv_actual)

    # outputformat inferred from filename
    s3_utils.write_dataframe(
        dataframe=CSV_EXPECTED, bucket=bucket.name, key=CSV_FILE_KEY
    )
    data_buffer = s3_utils._read_object(bucket=bucket.name, key=CSV_FILE_KEY)
    csv_actual = pd.read_csv(data_buffer)
    pd.testing.assert_frame_equal(CSV_EXPECTED, csv_actual)


def test_write_dataframe_tsv(bucket: Bucket) -> None:
    """Tests write_dataframe for a tsv file."""
    # outputformat given
    s3_utils.write_dataframe(
        dataframe=TSV_EXPECTED, bucket=bucket.name, key=TSV_FILE_KEY, outputformat="tsv"
    )
    data_buffer = s3_utils._read_object(bucket=bucket.name, key=TSV_FILE_KEY)
    tsv_actual = pd.read_csv(data_buffer, sep="\t")
    pd.testing.assert_frame_equal(TSV_EXPECTED, tsv_actual)

    # outputformat inferred from filename
    s3_utils.write_dataframe(
        dataframe=TSV_EXPECTED, bucket=bucket.name, key=TSV_FILE_KEY
    )
    data_buffer = s3_utils._read_object(bucket=bucket.name, key=TSV_FILE_KEY)
    tsv_actual = pd.read_csv(data_buffer, sep="\t")
    pd.testing.assert_frame_equal(TSV_EXPECTED, tsv_actual)


def test_write_dataframe_txt(bucket: Bucket) -> None:
    """Tests write_dataframe for a txt file."""
    # outputformat given
    s3_utils.write_dataframe(
        dataframe=TXT_EXPECTED, bucket=bucket.name, key=TXT_FILE_KEY, outputformat="txt"
    )
    data_buffer = s3_utils._read_object(bucket=bucket.name, key=TXT_FILE_KEY)
    txt_actual = pd.read_csv(data_buffer, sep="\t")
    pd.testing.assert_frame_equal(TXT_EXPECTED, txt_actual)

    # outputformat inferred from filename
    s3_utils.write_dataframe(
        dataframe=TXT_EXPECTED, bucket=bucket.name, key=TXT_FILE_KEY
    )
    data_buffer = s3_utils._read_object(bucket=bucket.name, key=TXT_FILE_KEY)
    txt_actual = pd.read_csv(data_buffer, sep="\t")
    pd.testing.assert_frame_equal(TXT_EXPECTED, txt_actual)


def test_write_numpy_array(bucket: Bucket) -> None:
    """Tests write_numpy_array."""
    s3_utils.write_numpy_array(
        array=NP_ARRAY_EXPECTED, bucket=bucket.name, key=NP_ARRAY_FILE_KEY
    )
    data = s3_utils._read_object(bucket=bucket.name, key=NP_ARRAY_FILE_KEY)
    np_array_actual = pickle.load(data)
    np.testing.assert_equal(np_array_actual, NP_ARRAY_EXPECTED)


def test_write_joblib(bucket: Bucket) -> None:
    """Tests write_joblib."""
    s3_utils.write_joblib(
        model=JOBLIB_EXPECTED, bucket=bucket.name, key=JOBLIB_FILE_KEY
    )
    data = s3_utils._read_object(bucket=bucket.name, key=JOBLIB_FILE_KEY)
    joblib_actual = joblib.load(data)

    assert joblib_actual == JOBLIB_EXPECTED


def test_write_json(bucket: Bucket) -> None:
    """Tests write_json."""
    s3_utils.write_json(dict_obj=JSON_EXPECTED, bucket=bucket.name, key=JSON_FILE_KEY)
    data_buffer = s3_utils._read_object(bucket=bucket.name, key=JSON_FILE_KEY)
    json_actual = json.loads(data_buffer.read())
    TestCase().assertDictEqual(JSON_EXPECTED, json_actual)
