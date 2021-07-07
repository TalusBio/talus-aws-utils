from typing import Any
from typing import Iterable

import boto3
import pytest
from moto import mock_s3
from mypy_boto3_s3.service_resource import Bucket


@pytest.fixture
def env_vars(monkeypatch: Any) -> None:
    monkeypatch.setenv("PROJECT_BUCKET", "test_bucket")


@pytest.fixture
def bucket() -> Iterable[Bucket]:
    REGION = "us-west-2"
    with mock_s3():
        s3 = boto3.resource("s3")
        bucket = s3.Bucket("test_bucket")
        bucket.create(CreateBucketConfiguration={"LocationConstraint": REGION})

        yield bucket
