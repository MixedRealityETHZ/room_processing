import logging
import os
import time
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from segmentation import segment_all_objects, remove_png_texture_files, separate_instances, create_pointcloud_from_obj
from vrra_seg.api import VrraApi

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def get_all_file_paths(directory):
    # initializing empty file paths list
    file_paths = []
    file_names = []
    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
            if filename[-4:] == '.obj':
                file_names.append(filename[:-4])

    # returning all file paths
    return file_paths, file_names


def tasks(api, poll_interval):
    while True:
        try:
            task = api.queue_pop()
            if task is None:
                time.sleep(poll_interval)
            else:
                logger.info(f"processing task id = {task['id']}")
                logger.debug(task)
                yield task
        except Exception as e:
            logger.error("failed to fetch task", exc_info=e)
            time.sleep(poll_interval)


if __name__ == '__main__':
    base_url = "https://vrra.howyoung.dev"
    poll_interval = 10

    api = VrraApi(base_url)
    for task in tasks(api, poll_interval):
        try:
            with TemporaryDirectory() as tmp:
                cwd = Path(tmp)
                asset = api.get_asset(task["assetId"])
                with open(cwd / asset['name'], 'wb') as f:
                    api.download(asset['url'], f)
                with ZipFile(cwd / asset['name']) as zf:
                    zf.extractall(cwd / 'extracted')
                logger.debug(f"model asset: {asset}")

                file = cwd / 'extracted'
                # file = '/home/lizeth/Downloads/appscanner/room2d22_15_42_39/'
                scene_name_obj = task['path']
                segmented_ply_name = f'{task["name"]}.ply'

                hd_ply_name = file / f'{task["name"]}_hd.ply'
                ply_asset = api.get_asset(task["pointCloudAssetId"])
                with open(hd_ply_name, 'wb') as f:
                    api.download(ply_asset['url'], f)

                # TODO: segment the model in `file`
                # segment object
                CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                                'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                                'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                                'bathtub', 'otherfurniture')

                id_class = 5
                segs_file = file / 'segs'
                orig_texture = file / 'textured_output.jpg'
                os.mkdir(segs_file)
                shutil.copy(orig_texture, segs_file / 'textured_output.jpg')

                create_pointcloud_from_obj(str(file / scene_name_obj), str(file / segmented_ply_name))
                segment_all_objects(file, scene_name_obj, segmented_ply_name, CLASS_LABELS)
                remove_png_texture_files(segs_file)

                # TODO: create a zip archive of all segmented models with name `model_zip_file`

                directory = file / "segs"
                file_paths, file_names = get_all_file_paths(directory)
                model_zip_file = file / "segs.zip"
                with ZipFile(model_zip_file, 'w') as zip:
                    for file in file_paths:
                        zip.write(file, Path(file).name)

                # TODO: (optional) add a thumbnail for the room
                # thumbnail_file = "textured_output.jpg"
                # with open(thumbnail_file, "rb") as f:
                #     thumbnail_asset = api.create_asset(thumbnail_file, f)
                # logger.debug(f"added asset: {thumbnail_asset}")
                room = api.add_room({
                    "name": task["name"],
                    "thumbnailAssetId": None  # thumbnail_asset["id"]
                })
                logger.debug(f"added room: {room}")

                with open(model_zip_file, "rb") as f:
                    upload_asset = api.create_asset(model_zip_file.name, f)
                logger.debug(f"added asset: {upload_asset}")

                for model_file in file_names:
                    # TODO: (optional) add a thumbnail for the model
                    # thumbnail_file = ""
                    # with open(thumbnail_file, "rb") as f:
                    #     thumbnail_asset = api.create_asset(thumbnail_file, f)
                    # logger.debug(f"added asset: {thumbnail_asset}")

                    model_file_data = np.load(directory / (model_file + '.npy'), allow_pickle=True)
                    model_file_data = model_file_data.item()

                    p_min, p_max = model_file_data['min_bound'], model_file_data['max_bound']
                    model = api.add_model({
                        "name": model_file_data['name'],  # TODO: model name # table or chair
                        "assetId": upload_asset["id"],
                        "path": model_file + '.obj',
                        "bounds": {
                            "pMin": {"x": p_min[0], "y": p_min[1], "z": p_min[1]},
                            "PMax": {"x": p_max[0], "y": p_max[1], "z": p_max[1]}
                        },  # TODO: provide a bounding box of the object
                        "thumbnailAssetId": None  # thumbnail_asset["id"]
                    })
                    logger.debug(f"added model: {model}")

                    center = model_file_data['center']
                    obj = api.add_room_obj(room["id"], {
                        # TODO: provide translation, rotation and scale of the object
                        "translation": {"x": center[0], "y": center[1], "z": center[2]},
                        "rotation": {"x": 0, "y": 0, "z": 0, "w": 0},
                        "scale": {"x": 1, "y": 1, "z": 1},
                        "modelId": model["id"],
                        "movable": model_file_data['movable'],  # TODO: set to false if the object is wall, floor, etc.
                    })
                    logger.debug(f"added obj: {obj}")

                api.set_task_completed(task["id"], {"message": "segmentation is successful"})
                logger.info(f"task complete id = {task['id']}")

        except Exception as e:
            logger.error(f"error occurred when processing task id = {task['id']}", exc_info=e)
            api.set_task_failed(task["id"], {"message": str(e)})
