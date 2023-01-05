import logging
import sys
from pathlib import Path

from vrra_seg.api import VrraApi

logging.basicConfig(level='INFO')

logger = logging.getLogger(__name__)
if __name__ == '__main__':
    file, ply_file, name, path = sys.argv[1:]
    file = Path(file)
    ply_file = Path(ply_file)
    base_url = "https://vrra.howyoung.dev"
    poll_interval = 10

    api = VrraApi(base_url)
    with open(file, "rb") as f:
        upload_asset = api.create_asset(file.name, f)
    logger.info(f"added asset: {upload_asset}")

    with open(ply_file, "rb") as f:
        ply_upload_asset = api.create_asset(ply_file.name, f)
    logger.info(f"added asset: {ply_upload_asset}")

    task = api.queue_push({
        "assetId": upload_asset["id"],
        "pointCloudAssetId": ply_upload_asset["id"],
        "name": name,
        "path": path,
    })
    logger.info(f"added task: {task}")
