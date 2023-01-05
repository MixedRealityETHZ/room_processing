import os
import sys
import argparse
import numpy as np
from urllib.request import urlretrieve
import copy

try:
    import open3d as o3d
except ImportError:
    raise ImportError('Please install open3d with `pip install open3d`.')

import torch
import MinkowskiEngine as ME
from minkunet import MinkUNet34C
import matplotlib.pyplot as plt
import scipy.spatial as spatial

from indoor_utils import inference_segmentation, custom_draw_geometry_with_key_callback, segment_object,\
                            pointcloud_from_mesh, save_obj_data, create_pointcloud, load_file, get_colors

def visualize_obj(obj_file):
    mesh = o3d.io.read_triangle_mesh(obj_file, True)
    custom_draw_geometry_with_key_callback(mesh)
def segment_all_objects(file_path, obj_name, ply_name, CLASS_LABELS):
    # segment mesh
    print('Starting inference ...')
    [pred_classes_hd, coords_hd, seg_colors, texture_colors] = inference_segmentation(str(file_path / ply_name[:-4]) + '_hd.ply')
    # [pred_classes, coords, seg_colors, texture_colors] = inference_segmentation(str(file_path / ply_name))

    points_hd, texture_colors_hd, pcd_hd = load_file(str(file_path / ply_name[:-4]) + '_hd.ply')
    points, texture_colors, pcd = load_file(str(file_path / ply_name))

    # otherwise KDTree construction may run out of recursions
    leaf_size = 1000
    sys.setrecursionlimit(int(max(1000, round(points_hd.shape[0] / leaf_size))))
    kdtree = spatial.KDTree(points_hd, leaf_size)

    points_dist, points_ids = kdtree.query(x=points, k=1)
    pred_classes = np.array(pred_classes_hd)[points_ids]

    seg_pcd = create_pointcloud(points, get_colors(pred_classes) / 255)
    # o3d.visualization.draw_geometries([seg_pcd])

    # segment object
    for i in range(0, len(CLASS_LABELS)):
        id_class = i
        segmented_obj_file = file_path / 'segs' / (CLASS_LABELS[id_class - 1] + '_' + ply_name[:-4] + '.obj')
        segment_object(str(file_path / obj_name), str(segmented_obj_file), CLASS_LABELS, id_class, pred_classes)
        replace_mtl_texture_files(file_path, ply_name, CLASS_LABELS, id_class)

        if os.path.exists(segmented_obj_file):
            if CLASS_LABELS[id_class - 1] != 'wall' and CLASS_LABELS[id_class - 1] != 'floor':
                separate_instances(file_path, ply_name, CLASS_LABELS, id_class)
def replace_mtl_texture_files(file_path, ply_name, CLASS_LABELS, id_class):
    texture_file_path = file_path / 'segs' / (CLASS_LABELS[id_class - 1] + '_' + ply_name[:-4] + '.mtl')
    search_string = CLASS_LABELS[id_class - 1] + '_' + ply_name[:-4] + '_0.png'

    files_exits = os.path.isfile(texture_file_path)

    if files_exits:
        with open(texture_file_path, 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace(search_string, 'textured_output.jpg')

        # Write the file out again
        with open(texture_file_path, 'w') as file:
            file.write(filedata)

def separate_instances(file_path, ply_name, CLASS_LABELS, id_class):
    mesh_filename = str(file_path / 'segs' / (CLASS_LABELS[id_class - 1] + '_' + ply_name[:-4] + '.obj'))
    mesh = o3d.io.read_triangle_mesh(mesh_filename, True)
    # o3d.visualization.draw_geometries([mesh])
    pcd = pointcloud_from_mesh(mesh)
    points = np.array(pcd.points)
    labels = np.array(
        pcd.cluster_dbscan(eps=0.4, min_points=25, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])

    if max_label > 0:
        for i in range(0, max_label+1):
            tmp_mesh = copy.deepcopy(mesh)
            mask = labels != i
            tmp_mesh.remove_vertices_by_mask(mask)
            vertices = np.array(tmp_mesh.vertices)
            # o3d.visualization.draw_geometries([tmp_mesh])
            if vertices.shape[0] > 100:
                segment_filename = mesh_filename[:-4] + '_' + str(i) + '.obj'
                save_obj_data(CLASS_LABELS, id_class, tmp_mesh, segment_filename)
                o3d.io.write_triangle_mesh(segment_filename, tmp_mesh)
                replace_mtl_texture_files(file_path, ply_name[:-4] + '_' + str(i) + '.ply', CLASS_LABELS, id_class)
                o3d.visualization.draw_geometries([tmp_mesh])
        os.remove(mesh_filename)
        os.remove(mesh_filename[:-4] + '.mtl')
        os.remove(mesh_filename[:-4] + '.npy')
def create_pointcloud_from_obj(input_filename_obj, output_filename_ply):

    mesh = o3d.io.read_triangle_mesh(input_filename_obj, True)
    textured_pcd = pointcloud_from_mesh(mesh)

    #rotate the axis to z
    textured_pcd.rotate(mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)),
                  center=(0, 0, 0))

    o3d.io.write_point_cloud(output_filename_ply, textured_pcd)
    # o3d.visualization.draw_geometries([textured_pcd])

def remove_png_texture_files(file_path):
    list_files = os.listdir(file_path)

    for item in list_files:
        if item.endswith(".png"):
            os.remove(os.path.join(file_path, item))