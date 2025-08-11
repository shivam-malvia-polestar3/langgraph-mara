import os
import json
import boto3
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def get_credentials_from_secrets_manager() -> dict:
  """
  Fetches generic username/password credentials from AWS Secrets Manager.
  Uses env vars AWS_SECRET_NAME and AWS_REGION (with fallbacks).
  """
  secret_name = os.environ.get("AWS_SECRET_NAME")
  if not secret_name:
    logger.warning("Environment variable 'AWS_SECRET_NAME' is not set or empty. Using default secret name.")
    secret_name = "test/maritime-analytics-platform/account-service"

  region_name = os.environ.get("AWS_REGION")
  if not region_name:
    logger.warning("Environment variable 'AWS_REGION' is not set or empty. Using default region 'us-east-1'.")
    region_name = "us-east-1"

  logger.info(f"Using AWS secret: {secret_name} from region: {region_name}")

  session = boto3.session.Session()
  client = session.client("secretsmanager", region_name=region_name)

  try:
    response = client.get_secret_value(SecretId=secret_name)
    secret_string = response.get("SecretString")
    if not secret_string:
      raise ValueError("SecretString is empty")
    secret_data = json.loads(secret_string)
    return {
      "username": secret_data.get("username"),
      "password": secret_data.get("password"),
      "ais_username": secret_data.get("ais_username"),
      "ais_password": secret_data.get("ais_password")
    }
  except ClientError as e:
    logger.error(f"Error fetching secrets: {e}")
    raise RuntimeError("Failed to retrieve secrets") from e


def get_basic_auth_credentials_from_secrets_manager(secret_env_var: str = "ASSET_INFO_SECRET_NAME") -> tuple:
  """
  Fetches Basic Auth credentials (username, password) from AWS Secrets Manager.

  - Reads the secret name from the given env var (default: ASSET_INFO_SECRET_NAME).
  - Falls back to a default if not provided.
  - Returns: (username, password) tuple for direct use in requests.auth.

  Example:
      from secret_manager import get_basic_auth_credentials_from_secrets_manager
      username, password = get_basic_auth_credentials_from_secrets_manager()
      resp = requests.get(url, auth=(username, password))
  """
  secret_name = os.environ.get(secret_env_var)
  if not secret_name:
    logger.warning(f"Environment variable '{secret_env_var}' is not set or empty. Using default secret name.")
    secret_name = "test/asset-info/opens"

  region_name = os.environ.get("AWS_REGION")
  if not region_name:
    logger.warning("Environment variable 'AWS_REGION' is not set or empty. Using default region 'us-east-1'.")
    region_name = "us-east-1"

  logger.info(f"Using AWS Basic Auth secret: {secret_name} from region: {region_name}")

  session = boto3.session.Session()
  client = session.client("secretsmanager", region_name=region_name)

  try:
    response = client.get_secret_value(SecretId=secret_name)
    secret_string = response.get("SecretString")
    if not secret_string:
      raise ValueError("SecretString is empty")
    secret_data = json.loads(secret_string)

    username = secret_data.get("username")
    password = secret_data.get("password")

    if not username or not password:
      raise ValueError("Basic Auth secret missing 'username' or 'password' fields")

    return username, password

  except ClientError as e:
    logger.error(f"Error fetching Basic Auth secrets: {e}")
    raise RuntimeError("Failed to retrieve Basic Auth secrets") from e
