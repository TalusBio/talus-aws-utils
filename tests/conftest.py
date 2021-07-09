"""tests/conftest.py"""
from typing import Any, Iterable

import boto3
import pytest

from moto import mock_s3
from mypy_boto3_s3.service_resource import Bucket


@pytest.fixture
def env_vars(monkeypatch: Any) -> None:
    """Monkeypatch Environment variables.

    Args:
        monkeypatch (Any): monkeypatch package
    """
    monkeypatch.setenv("PROJECT_BUCKET", "test_bucket")


@pytest.fixture
def bucket() -> Iterable[Bucket]:
    """Create a bucket fixture.

    Returns:
        Iterable[Bucket]: Bucket Fixture

    Yields:
        Iterator[Iterable[Bucket]]: Bucket Fixture
    """
    with mock_s3():
        s3 = boto3.resource("s3")
        bucket = s3.Bucket("test_bucket")
        bucket.create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        yield bucket
