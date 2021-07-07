"""src/talus_aws_utils/s3.py"""
import json
import pathlib
from io import BytesIO
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from hurry.filesize import size


def _get_boto_session() -> boto3.Session:
    """Creates and returns an active boto3 session.

    Returns:
        boto3.Session: An active boto3 Session.
    """
    session = boto3.Session()
    return session


def _read_object(bucket: str, key: str) -> BytesIO:
    """Reads an object in byte format from a given s3 bucket and key name.

    Args:
        bucket (str): The S3 bucket to load from.
        key (str): The object key within the s3 bucket.

    Raises:
        ValueError: If the file couldn't be found.

    Returns:
        BytesIO: The object in byte format.
    """
    s3_resource = _get_boto_session().resource("s3")
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
    """Writes an object in byte format to a given S3 bucket using the given key name.

    Args:
        bucket (str): The S3 bucket to write to.
        key (str): The object key within the s3 bucket to write to.
        buffer (BytesIO): The BytesIO object containing the data to write.
    """
    s3_client = _get_boto_session().client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def read_dataframe(
    bucket: str, key: str, inputformat: Optional[str] = None, **kwargs
) -> pd.DataFrame:
    """Reads a pandas dataframe from a given s3 bucket and key.
    An input format can be manually specified. Otherwise the
    function will try to infer it from the given object key.

    Args:
        bucket (str): The S3 bucket to load from.
        key (str): The object key within the s3 bucket.
        inputformat (Optional[str], optional): The target inputformat.
                                               Can be one of {parquet, txt, csv, tsv}.
                                               Defaults to None.
        kwargs (Dict, optional): Additional keyword arguments.

    Raises:
        ValueError: If either an incorrect inputformat is given or inferred
                    when None is given.

    Returns:
        pd.DataFrame: A pandas DataFrame.
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
    **kwargs,
) -> None:
    """Writes a pandas dataframe to a given s3 bucket using the given key.
    An output format can be manually specified. Otherwise the
    function will try to infer it from the given object key.

    Args:
        dataframe (pd.DataFrame): The pandas DataFrame to write.
        bucket (str): The S3 bucket to write to.
        key (str): The object key within the s3 bucket to write to.
        outputformat (Optional[str], optional): The target output format.
                                                Can be one of {parquet, txt, csv, tsv}.
                                                Defaults to None.
        kwargs (Dict, optional): Additional keyword arguments.

    Raises:
        ValueError: If either an incorrect inputformat is given or inferred
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


def read_json(bucket: str, key: str) -> Union[Any, Dict]:
    """Reads a json object from a given s3 bucket and key.

    Args:
        bucket (str): The S3 bucket to load from.
        key (str): The object key within the s3 bucket.

    Returns:
        Dict: A Python Dict of the loaded json object.
    """
    file_content = _read_object(bucket=bucket, key=key)
    return json.loads(file_content.read())


def write_json(dict_obj: Dict[str, Any], bucket: str, key: str) -> None:
    """Write a Dict to S3 as a json file.

    Args:
        dict_obj (Dict): The Dict object to save as json.
        bucket (str): The S3 bucket to write to.
        key (str): The object key within the s3 bucket to write to.
    """
    buffer = BytesIO()
    buffer.write(json.dumps(dict_obj).encode("utf-8"))
    buffer.seek(0)
    _write_object(bucket=bucket, key=key, buffer=buffer)


def file_keys_in_bucket(
    bucket: str, key: str, file_type: str = ""
) -> List[Optional[str]]:
    """Gets all the file keys in a given bucket, return empty list if none exist.

    Args:
        bucket (str): The S3 bucket to load from.
        key (str): The object key within the s3 bucket.
        file_type (str): A specific file type we want
                                   to filter for. Defaults to "".

    Returns:
        List[Optional[str]]: A List of S3 file keys.
    """
    s3_client = _get_boto_session().client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key)
    contents = response.get("Contents", [])

    return [obj.get("Key") for obj in contents if obj["Key"].endswith(file_type)]


def file_exists_in_bucket(bucket: str, key: str) -> bool:
    """Checks whether a file key exists in bucket.

    Args:
        bucket (str): The S3 bucket to load from.
        key (str): The object key within the s3 bucket.

    Raises:
        ClientError: If boto3 fails to retrieve the file metadata.

    Returns:
        bool: True if the file key exists, False if it doesn't.
    """
    s3_client = _get_boto_session().client("s3")
    try:
        _ = s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise e


def file_size(bucket: str, key: str, raw_size: bool = False) -> Union[str, Any]:
    """Gets the size for a file with key in given bucket.

    Args:
        bucket (str): The S3 bucket to load from.
        key (str): The object key within the s3 bucket.
        raw_size (bool): If True, returns the raw content length.
                         If False, returns a human-readable version e.g. 1KB.

    Raises:
        ValueError: If file doesn't exist.

    Returns:
        str: A str containing the file size.
    """
    s3_client = _get_boto_session().client("s3")
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
