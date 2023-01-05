
import os
import numpy as np
from pathlib import Path

from zipfile import ZipFile
from segmentation import segment_all_objects, remove_png_texture_files, visualize_obj, separate_instances, create_pointcloud_from_obj
from indoor_utils import reconstruct_mesh_from_pointcloud_with_texture, segment_object


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

if __name__ == '__main__':

    file_path = '/home/lizeth/Downloads/appscanner/room2d22_15_42_39/'
    obj_name = 'textured_output.obj'
    ply_name = 'room2d22.ply'
    seg_obj_name = 'wall_room2d22.obj'

    # segment object
    CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                    'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                    'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                    'bathtub', 'otherfurniture')


    #visualize
    id_class = 4
    segmented_obj_file = file_path + 'segs/' + CLASS_LABELS[id_class - 1] + '_' + ply_name[:-4] +'.obj'

    # visualize_obj(segmented_obj_file)
    visualize_obj(file_path + obj_name)


    directory = file_path + "segs"
    file_paths, model_files  = get_all_file_paths(directory)

    create_pointcloud_from_obj(str(Path(file_path)/obj_name), str(Path(file_path)/ply_name))
    segment_all_objects(Path(file_path), obj_name, ply_name, CLASS_LABELS)
    # remove_png_texture_files(Path(file_path) / 'segs/')
    #
    # reconstruct_mesh_from_pointcloud_with_texture(segmented_obj_file)

