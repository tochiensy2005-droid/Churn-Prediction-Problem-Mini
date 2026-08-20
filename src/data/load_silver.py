"""src/data/load_silver.py

Handles loading silver tables from either MinIO object storage or local disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from minio import Minio
from src import config

logger = logging.getLogger(__name__)


import socket

def is_minio_reachable(endpoint: str) -> bool:
    try:
        if ":" in endpoint:
            host, port_str = endpoint.split(":", 1)
            port = int(port_str)
        else:
            host = endpoint
            port = 9000
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except Exception:
        return False


def get_minio_client() -> Minio | None:
    endpoint = config.MINIO_ENDPOINT
    access_key = config.MINIO_ACCESS_KEY
    secret_key = config.MINIO_SECRET_KEY
    if not endpoint or not access_key:
        return None
    if not is_minio_reachable(endpoint):
        return None
    try:
        return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
    except Exception as e:
        logger.warning(f"Could not connect to MinIO at {endpoint}: {e}")
        return None


def load_silver_table(table_name: str) -> pd.DataFrame:
    """Load a silver table from MinIO or fallback to local disk."""
    client = get_minio_client()
    if client:
        try:
            if client.bucket_exists(config.MINIO_BUCKET):
                logger.info(f"Loading {table_name} from MinIO bucket {config.MINIO_BUCKET}")
                temp_path = config.OUTPUT_DIR / f"temp_{table_name}.parquet"
                
                # Fetch objects under the prefix table_name/
                objects = client.list_objects(config.MINIO_BUCKET, prefix=f"{table_name}/", recursive=True)
                files = [obj.object_name for obj in objects if obj.object_name.endswith(".parquet")]
                
                if files:
                    dfs = []
                    for f in files:
                        client.fget_object(config.MINIO_BUCKET, f, str(temp_path))
                        dfs.append(pd.read_parquet(temp_path))
                    if temp_path.exists():
                        temp_path.unlink()
                    return pd.concat(dfs, ignore_index=True)
        except Exception as e:
            logger.warning(f"Failed to fetch {table_name} from MinIO, falling back to local files: {e}")
            
    # Local fallback
    local_path = config.DATA_DIR / table_name
    logger.info(f"Loading {table_name} from local path: {local_path}")
    
    if local_path.is_file():
        return pd.read_parquet(local_path)
        
    files = sorted(local_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for {table_name} in {local_path}")
        
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
