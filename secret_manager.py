import os
import json
import boto3
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def get_credentials_from_secrets_manager() -> dict:
  # Read from env, fallback only if not set or empty
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
      "password": secret_data.get("password")
    }
  except ClientError as e:
    logger.error(f"Error fetching secrets: {e}")
    raise RuntimeError("Failed to retrieve secrets") from e