"""src/talus_aws_utils/lambda_.py module."""
import json

from typing import Any, Dict


def error_msg_wrapper(msg: str) -> Dict[str, Any]:
    """Wrap an error message in a JSON response.

    Returns
    -------
    Dict[str: Any]
        A JSON response.
    """
    return {
        "statusCode": 500,
        "headers": {
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
        },
        "body": json.dumps(msg),
    }


def success_msg_wrapper(msg: str) -> Dict[str, Any]:
    """Wrap a success message in a JSON response.

    Returns
    -------
    Dict[str: Any]
        A JSON response.
    """
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
        },
        "body": json.dumps(msg),
    }
