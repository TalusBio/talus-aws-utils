"""src/talus_aws_utils/s3.py module."""
import json
import os
import pathlib
import pickle

from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import boto3
import joblib
import numpy as np
import pandas as pd

from botocore.exceptions import ClientError
from hurry.filesize import size


def _read_object(bucket: str, key: str) -> BytesIO:
    """Read an object in byte format from a given s3 bucket and key name.

    Parameters
    ----------
    bucket : str
        The S3 bucket to load from.
    key : str
        The object key within the s3 bucket.

    Returns
    -------
    BytesIO
        The object in byte format.

    Raises
    ------
    ValueError
        If the file couldn't be found.

    """
    s3_resource = boto3.Session().resource("s3")
    s3_bucket = s3_resource.Bucket(bucket)
    data = BytesIO()
    try:
        s3_bucket.download_fileobj(Key=key, Fileobj=data)
        data.seek(0)
        return data
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise ValueError("File doesn't exist.")
        else:
            raise


def _write_object(bucket: str, key: str, buffer: BytesIO) -> None:
    """Write an object in byte format to a given S3 bucket using the given key name.

    Parameters
    ----------
    bucket : str
        The S3 bucket to write to.
    key : str
        The object key within the s3 bucket to write to.
    buffer : BytesIO
        The BytesIO object containing the data to write.

    """
    s3_client = boto3.Session().client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def read_dataframe(
    bucket: str, key: str, inputformat: Optional[str] = None, **kwargs: str
) -> pd.DataFrame:
    """Read a pandas dataframe from a given s3 bucket and key.
    An input format can be manually specified. Otherwise the
    function will try to infer it from the given object key.

    Parameters
    ----------
    bucket : str
        The S3 bucket to load from.
    key : str
        The object key within the s3 bucket.
    inputformat : Optional[str]
        The target inputformat.
        Can be one of {parquet, txt, csv, tsv}.
        (Default value = None).
    kwargs : Dict
        Additional keyword arguments.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame.

    Raises
    ------
    ValueError
        If either an incorrect inputformat is given or inferred
        when None is given.

    """
    if not inputformat:
        inputformat = pathlib.Path(key).suffix[1:]

    data = _read_object(bucket=bucket, key=key)

    if inputformat == "parquet":
        return pd.read_parquet(data, **kwargs)
    elif inputformat == "csv":
        return pd.read_csv(data, **kwargs)
    elif inputformat == "tsv" or inputformat == "txt":
        return pd.read_csv(data, sep="\t", **kwargs)
    else:
        raise ValueError(
            "Invalid (inferred) inputformat. Use one of: parquet, txt, csv, tsv."
        )


def write_dataframe(
    dataframe: pd.DataFrame,
    bucket: str,
    key: str,
    outputformat: Optional[str] = None,
    **kwargs: str,
) -> None:
    """Write a pandas dataframe to a given s3 bucket using the given key.
    An output format can be manually specified. Otherwise the
    function will try to infer it from the given object key.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame to write.
    bucket : str
        The S3 bucket to write to.
    key : str
        The object key within the s3 bucket to write to.
    outputformat : Optional[str]
        The target output format.
        Can be one of {parquet, txt, csv, tsv}.
        (Default value = None).
    kwargs : Dict
        Additional keyword arguments.

    Raises
    ------
    ValueError
        If either an incorrect inputformat is given or inferred
        when None is given.

    """
    if not outputformat:
        outputformat = pathlib.Path(key).suffix[1:]

    buffer = BytesIO()
    if outputformat == "parquet":
        dataframe.to_parquet(buffer, engine="pyarrow", index=False, **kwargs)
    elif outputformat == "csv":
        dataframe.to_csv(buffer, index=False, **kwargs)
    elif outputformat == "tsv" or outputformat == "txt":
        dataframe.to_csv(buffer, sep="\t", index=False, **kwargs)
    else:
        raise ValueError(
            "Invalid (inferred) outputformat. Use one of: parquet, txt, csv, tsv."
        )
    _write_object(bucket=bucket, key=key, buffer=buffer)


def read_numpy_array(
    bucket: str,
    key: str,
) -> Any:
    """Read a numpy array from a given s3 bucket and key.

    Parameters
    ----------
    bucket : str
        The S3 bucket to load from.
    key : str
        The object key within the s3 bucket.

    Returns
    -------
    np.array
        A numpy array.

    """
    data = _read_object(bucket=bucket, key=key)
    return np.load(data, allow_pickle=True)


def write_numpy_array(
    array: np.array,
    bucket: str,
    key: str,
) -> None:
    """Write a numpy array to a given s3 bucket using the given key.

    Parameters
    ----------
    array : np.array
        The numpy array to write.
    bucket : str
        The S3 bucket to write to.
    key : str
        The object key within the s3 bucket to write to.

    """
    buffer = BytesIO()
    pickle.dump(array, buffer)
    buffer.seek(0)
    _write_object(bucket=bucket, key=key, buffer=buffer)


def read_joblib(
    bucket: str,
    key: str,
) -> Any:
    """Read a joblib model from a given s3 bucket and key.

    Parameters
    ----------
    bucket : str
        The S3 bucket to load from.
    key : str
        The object key within the s3 bucket.

    Returns
    -------
    np.array
        A joblib model.

    """
    data = _read_object(bucket=bucket, key=key)
    return joblib.load(data)


def write_joblib(
    model: Any,
    bucket: str,
    key: str,
) -> None:
    """Write a joblib model to a given s3 bucket using the given key.

    Parameters
    ----------
    array : np.array
        The joblib model to write.
    bucket : str
        The S3 bucket to write to.
    key : str
        The object key within the s3 bucket to write to.

    """
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    _write_object(bucket=bucket, key=key, buffer=buffer)


def read_json(bucket: str, key: str) -> Union[Any, Dict[str, Any]]:
    """Read a json object from a given s3 bucket and key.

    Parameters
    ----------
    bucket : str
        The S3 bucket to load from.
    key : str
        The object key within the s3 bucket.

    Returns
    -------
    Dict
        A Python Dict of the loaded json object.

    """
    file_content = _read_object(bucket=bucket, key=key)
    return json.loads(file_content.read())


def write_json(dict_obj: Dict[str, Any], bucket: str, key: str) -> None:
    """Write a Dict to S3 as a json file.

    Parameters
    ----------
    dict_obj : Dict[str, Any]
        The Dict object to save as json.
    bucket : str
        The S3 bucket to write to.
    key : str
        The object key within the s3 bucket to write to.

    """
    buffer = BytesIO()
    buffer.write(json.dumps(dict_obj).encode("utf-8"))
    buffer.seek(0)
    _write_object(bucket=bucket, key=key, buffer=buffer)


def file_keys_in_bucket(
    bucket: str, key: str, file_type: Optional[str] = ""
) -> List[Optional[str]]:
    """Get all the file keys in a given bucket, return empty list if none exist.

    Parameters
    ----------
    bucket : str
        The S3 bucket to load from.
    key : str
        The object key within the s3 bucket.
    file_type : str
        A specific file type we want
        to filter for. (Default value = "").

    Returns
    -------
    List[Optional[str]]
        A List of S3 file keys.

    """
    s3_client = boto3.Session().client("s3")
    keys_left = True
    keys = []
    while keys_left:
        start_tkn = keys[-1] if keys else ""
        response = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=key, StartAfter=start_tkn
        )
        contents = response.get("Contents", [])
        keys_left = contents != []
        keys += [obj.get("Key") for obj in contents]

    return [k for k in keys if os.path.splitext(k)[1] and k.endswith(file_type)]


def file_exists_in_bucket(bucket: str, key: str) -> bool:
    """Check whether a file key exists in bucket.

    Parameters
    ----------
    bucket : str
        The S3 bucket to load from.
    key : str
        The object key within the s3 bucket.

    Returns
    -------
    bool
        True if the file key exists, False if it doesn't.

    Raises
    ------
    ClientError
        If boto3 fails to retrieve the file metadata.

    """
    s3_client = boto3.Session().client("s3")
    try:
        _ = s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise e


def file_size(bucket: str, key: str, raw_size: bool = False) -> Union[str, Any]:
    """Get the size for a file with key in given bucket.

    Parameters
    ----------
    bucket : str
        The S3 bucket to load from.
    key : str
        The object key within the s3 bucket.
    raw_size : bool
        If True, returns the raw content length.
        If False, returns a human-readable version e.g. 1KB.
        (Default value = False).

    Returns
    -------
    str
        A str containing the file size.

    Raises
    ------
    ValueError
        If file doesn't exist.

    """
    s3_client = boto3.Session().client("s3")
    try:
        file = s3_client.head_object(Bucket=bucket, Key=key)
        content_length = file["ContentLength"]
        if raw_size:
            return str(content_length)
        else:
            return size(content_length)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise ValueError("File doesn't exist. Couldn't retrieve file size.")
        else:
            raise
