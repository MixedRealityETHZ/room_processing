# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import argparse
import numpy as np
from urllib.request import urlretrieve
import copy
import trimesh
import matplotlib.pyplot as plt

try:
    import open3d as o3d
except ImportError:
    raise ImportError('Please install open3d with `pip install open3d`.')

import torch
import MinkowskiEngine as ME
from minkunet import MinkUNet34C

# Check if the weights and file exist and download
# if not os.path.isfile('weights.pth'):
#     print('Downloading weights...')
#     urlretrieve("https://bit.ly/2O4dZrz", "weights.pth")

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='/mnt/DATA/data_indoor/appscanner/my_office_22_22_14_axis_z.ply')
parser.add_argument('--weights', type=str, default='weights.pth')
parser.add_argument('--use_cpu', action='store_true')

CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                'bathtub', 'otherfurniture')

VALID_CLASS_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
]

SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.), #chair
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}


def load_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors)
    return coords, colors, pcd


def normalize_color(color: torch.Tensor, is_color_in_range_0_255: bool = False) -> torch.Tensor:
    r"""
    Convert color in range [0, 1] to [-0.5, 0.5]. If the color is in range [0,
    255], use the argument `is_color_in_range_0_255=True`.

    `color` (torch.Tensor): Nx3 color feature matrix
    `is_color_in_range_0_255` (bool): If the color is in range [0, 255] not [0, 1], normalize the color to [0, 1].
    """
    if is_color_in_range_0_255:
        color /= 255
    color -= 0.5
    return color.float()

def inference_segmentation(input_filename_ply):
    config = parser.parse_args()
    config.file_name = input_filename_ply
    device = torch.device('cuda' if (
            torch.cuda.is_available() and not config.use_cpu) else 'cpu')
    print(f"Using {device}")
    # Define a model and load the weights
    model = MinkUNet34C(3, 20).to(device)
    model_dict = torch.load(config.weights)
    model.load_state_dict(model_dict)
    model.eval()

    coords, texture_colors, pcd = load_file(config.file_name)
    # Measure time
    with torch.no_grad():
        voxel_size = 0.02
        # Feed-forward pass and get the prediction
        in_field = ME.TensorField(
            features=normalize_color(torch.from_numpy(texture_colors)),
            coordinates=ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=device,
        )
        # Convert to a sparse tensor
        sinput = in_field.sparse()
        # Output sparse tensor
        soutput = model(sinput)
        # get the prediction on the input tensor field
        out_field = soutput.slice(in_field)
        logits = out_field.F

    _, pred = logits.max(1)
    pred = pred.cpu().numpy()


    # Map color
    colors = np.array([SCANNET_COLOR_MAP[VALID_CLASS_IDS[l]] for l in pred])

    #class_index = np.array( [i  for i in range(0, pred.shape[0]) if VALID_CLASS_IDS[ pred[i]] == id_class] )
    pred_classes = [VALID_CLASS_IDS[l] for l in pred]

    return pred_classes, coords, colors, texture_colors

def get_colors(pred):
    colors = np.array([SCANNET_COLOR_MAP[l] for l in pred])
    return colors
def save_object(class_index, coords, colors, save_file):
    colors_obj = colors[class_index]
    coord_obj = coords[class_index]

    obj_pcd = create_pointcloud(coord_obj, colors_obj)
    o3d.io.write_point_cloud(save_file, obj_pcd)

    o3d.visualization.draw_geometries([obj_pcd])

def create_pointcloud(coords, colors):
    # Create a point cloud file
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(coords)
    pred_pcd.colors = o3d.utility.Vector3dVector(colors)
    pred_pcd.estimate_normals()

    return pred_pcd

def pointcloud_from_mesh(mesh):
    textures = np.asarray(mesh.textures[0])
    triangles = np.array(mesh.triangles)
    triangle_uvs = np.array(mesh.triangle_uvs)
    texture_side_length = textures.shape[0]
    triangle_uvs_img = (np.floor(triangle_uvs * (texture_side_length - 1))).astype(int)
    color_uvs = textures[triangle_uvs_img[:, 1], triangle_uvs_img[:, 0]]
    triangles_indexes = triangles.reshape(-1)
    n_vertices = np.array(mesh.vertices).shape[0]
    vertex_colors = np.zeros((n_vertices, 3)).astype(float)

    for i in range(0, color_uvs.shape[0]):
        vertex_colors[triangles_indexes[i]] = color_uvs[i]

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    #o3d.visualization.draw_geometries([mesh])

    textured_pcd = o3d.geometry.PointCloud()
    textured_pcd.points = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    textured_pcd.colors = o3d.utility.Vector3dVector(vertex_colors / 255)
    textured_pcd.estimate_normals()

    return textured_pcd

def rotate_coordinate_frame_test(input_filename_obj):

    mesh = o3d.io.read_triangle_mesh(input_filename_obj, True)
    mesh_r = copy.deepcopy(mesh).translate((10, 0, 0))
    mesh_r.rotate(mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0)),
                  center=(0, 0, 0))
    o3d.visualization.draw_geometries([mesh, mesh_r])

def save_obj_data(CLASS_LABELS, id_class, obj_mesh, segment_filename):
    obj_data = {}
    obj_data["name"] = CLASS_LABELS[id_class - 1]
    obj_data["center"] = obj_mesh.get_center()
    obj_data["movable"] = True
    if obj_data["name"] == 'wall' or obj_data["name"] == 'floor' or obj_data["name"] == 'window' or obj_data["name"] == 'door':
        obj_data["movable"] = False
    obj_mesh.translate(-obj_mesh.get_center())
    obj_data["max_bound"] = obj_mesh.get_max_bound()
    obj_data["min_bound"] = obj_mesh.get_min_bound()
    np.save(segment_filename[:-4] + '.npy' , obj_data)

def segment_object(obj_filename, segment_filename, CLASS_LABELS,  id_class, pred_classes):

    mesh = o3d.io.read_triangle_mesh(obj_filename, True)
    obj_mesh = copy.deepcopy(mesh)

    mask = np.array(pred_classes) != id_class
    obj_mesh.remove_vertices_by_mask(mask)

    vertices = np.array(obj_mesh.vertices)
    # o3d.visualization.draw_geometries([obj_mesh])
    if vertices.shape[0] > 100:
        vertices = np.array(obj_mesh.vertices)
        if vertices.shape[0] > 300:
            save_obj_data(CLASS_LABELS, id_class, obj_mesh, segment_filename)
            o3d.io.write_triangle_mesh(segment_filename[:-4] + '.obj', obj_mesh)
            # o3d.visualization.draw_geometries([obj_mesh])

def read_rotate_save_axis_z_pointcloud(pc_filename):

    pcd = o3d.io.read_point_cloud(pc_filename)
    # rotate the axis to z
    pcd.rotate(pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)),
                        center=(0, 0, 0))
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(pc_filename[:-4] + '_z_axis.ply', pcd)

def visualize_mesh_pointcloud(mesh_filename, pcd_filename):

    mesh = o3d.io.read_triangle_mesh(mesh_filename, True)
    pcd = o3d.io.read_point_cloud(pcd_filename)
    o3d.visualization.draw_geometries([mesh, pcd])

def rayTriangleIntersect(orig, dir, N, v0, v1, v2):

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    N = np.cross(v0v1, v0v2)
    epsilon = 0.0000001
    NdotRayDirection = np.dot(N, dir)
    if np.abs(NdotRayDirection) < epsilon: return False #they are parallel

    d = np.dot(-N, v0)
    t = -(np.dot(-N, orig) + d) / NdotRayDirection
    if t < 0: return False # the triangle is behind

    # intersection point
    P = orig + t * dir

    edge0 = v1 - v0
    vp0 = P - v0
    C = np.cross(edge0, vp0)
    if np.dot(N, C) < 0: return False # P is on the right side

    edge1 = v2 - v1
    vp1 = P - v1
    C = np.cross(edge1, vp1)
    if np.dot(N, C) < 0: return False

    edge2 = v0 - v2
    vp2 = P - v2
    C = np.cross(edge2, vp2)
    if np.dot(N, C) < 0: return False

    return True

def reconstruct_mesh_from_pointcloud_with_texture(mesh_filename):

    mesh = o3d.io.read_triangle_mesh(mesh_filename, True)
    pcd = pointcloud_from_mesh(mesh)
    # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    # o3d.visualization.draw_geometries([cl])

    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    # rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)
    # o3d.visualization.draw_geometries([rec_mesh])

    (convex_hull_mesh, pts_indexes) = o3d.geometry.PointCloud.compute_convex_hull(pcd, True)

    points = np.array(pcd.points)
    normals = np.array(pcd.normals)
    triangles = np.array(convex_hull_mesh.triangles)
    vertices = np.array(convex_hull_mesh.vertices)
    convex_hull_mesh.compute_triangle_normals(True)
    triangle_normals = np.array(convex_hull_mesh.triangle_normals)

    convex_trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_normals=triangle_normals)
    # new_vertices, new_triangles = trimesh.remesh.subdivide(vertices=vertices, faces=triangles)
    # convex_trimesh = trimesh.Trimesh(vertices=new_vertices, faces=new_triangles)
    # convex_trimesh.export("/home/lizeth/Downloads/debug/convex_filtered.ply")
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(convex_trimesh)
    index_tri, index_ray = intersector.intersects_id(points, normals, False, 1, False)
    unique, counts = np.unique(index_tri, return_counts=True)
    # sorted_counts_index = sorted(range(len(counts)), key=lambda k: counts[k])
    triangles_area = trimesh.triangles.area(triangles=vertices[triangles])
    sorted_area_index = sorted(range(len(triangles_area)), key=lambda k: triangles_area[k])

    # convex_hull_mesh.remove_triangles_by_index(unique)
    # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(convex_hull_mesh)
    # hull_ls.paint_uniform_color((1, 0, 0))
    # o3d.visualization.draw_geometries([pcd, hull_ls])
    o3d.visualization.draw_geometries([mesh])
    mesh_triangles = np.array(mesh.triangles)
    for i in reversed(sorted_area_index):
        #i is the largest
        # convex_hull_mesh.remove_triangles_by_index([i])
        triangles_area[i]
        #add triangle
        triangles[i]
        pts_indexes
        new_triangle = [pts_indexes[triangles[i][0]], pts_indexes[triangles[i][1]], pts_indexes[triangles[i][2]]]
        new_triangles_mesh = np.concatenate((mesh_triangles, [new_triangle]), axis=0)
        mesh.triangles = o3d.utility.Vector3iVector(new_triangles_mesh)
        o3d.visualization.draw_geometries([mesh])

        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(convex_hull_mesh)
        hull_ls.paint_uniform_color((1, 0, 0))
        o3d.visualization.draw_geometries([pcd, hull_ls])


    #filtered_index_tri = [unique[unique[i]] for i in range(len(unique)) if triangles_area[unique[i]]/counts[unique[i]]]
    # unique = list(unique)
    # area_count_ratio = [triangles_area[unique[i]]/counts[i] for i in range(0, len(unique))]
    # sorted_ratio_index = sorted(range(len(area_count_ratio)), key=lambda k: area_count_ratio[k])

    # filtered_index_tri = [unique[sorted_ratio_index[i]] for i in range(len(sorted_ratio_index)) if  area_count_ratio[sorted_ratio_index[i]] > 0.0001]
    convex_hull_mesh.remove_triangles_by_index(unique)
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(convex_hull_mesh)
    hull_ls.paint_uniform_color((1, 0, 0))
    o3d.visualization.draw_geometries([pcd, hull_ls])

    o3d.io.write_triangle_mesh(mesh_filename[:-4] + '_convex_filtered.ply', convex_hull_mesh)

    #TODO ray casting with point normals to convex hull mesh
    # sampling points in the rest of convex hull mesh
    #apply poisson

    # o3d.geometry.PointCloud.compute_point_cloud_distance

    # #Mapping right textures to triangles uv
    # triangle_uvs = np.random.rand(np.array(rec_mesh.triangles).shape[0] * 3, 2)
    # rec_mesh.textures = mesh.textures
    # rec_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)

    # o3d.visualization.draw_geometries([rec_mesh])
    # o3d.io.write_triangle_mesh(mesh_filename[:-4] + "_rec.obj", rec_mesh)

    print('Reconstruction done!')

def custom_draw_geometry_with_key_callback(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    key_to_callback = {}
    key_to_callback[ord("B")] = change_background_to_black
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)















