"""src/talus_aws_utils/secrets.py module."""
import base64
import json

from typing import Dict

import boto3

from botocore.exceptions import ClientError


def get_secret(secret_name: str, region_name: str) -> Dict[str, str]:
    """Get a secret value from AWS Secret Manager.

    Parameters
    ----------
    secret_name : str
        Name of the secret to get
    region_name : str
        Name of the region to get the secret from

    Returns
    -------
    Dict[str, str]
        The secret value


    Raises
    ------
    ClientError
        If the secret is not found
    """
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if "SecretString" in get_secret_value_response:
            secret = get_secret_value_response["SecretString"]
        else:
            secret = base64.b64decode(get_secret_value_response["SecretBinary"])

    return json.loads(secret)
