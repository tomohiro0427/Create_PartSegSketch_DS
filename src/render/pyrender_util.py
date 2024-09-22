import numpy as np
from src.render.render_utils import Render, create_pose, create_shapenet_chair_camera_pose
# from ..sketch_utils import create_random_pose
import matplotlib
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

matplotlib.use("Agg")
import pyglet
import trimesh
import open3d as o3d



# FrontVector = (np.array([[0.52573, 0.38197, 0.85065],
#                          [-0.20081, 0.61803, 0.85065],
#                          [-0.64984, 0.00000, 0.85065],
#                          [-0.20081, -0.61803,  0.85065],
#                          [0.52573, -0.38197, 0.85065],
#                          [0.85065, -0.61803, 0.20081],
#                          [1.0515,  0.00000, -0.20081],
#                          [0.85065, 0.61803, 0.20081],
#                          [0.32492, 1.00000, -0.20081],
#                          [-0.32492, 1.00000,  0.20081],
#                          [-0.85065, 0.61803, -0.20081],
#                          [-1.0515, 0.00000,  0.20081],
#                          [-0.85065, -0.61803, -0.20081],
#                          [-0.32492, -1.00000,  0.20081],
#                          [0.32492, -1.00000, -0.20081],
#                          [0.64984, 0.00000, -0.85065],
#                          [0.20081, 0.61803, -0.85065],
#                          [-0.52573, 0.38197, -0.85065],
#                          [-0.52573, -0.38197, -0.85065],
#                          [0.20081, -0.61803, -0.85065]]))

# FrontVector = (np.array([
#                          [0.8, 0.0, -0.8],
#                          [-0.8, 0.0, -0.8],
#                          [0.8, 0.5, -0.8],
#                          [-0.8, 0.5, -0.8],
#                          [1.2, 0.0, 0],
#                          [-1.2, 0.0, 0],
#                          [0, 0.0, -1.2],
#                          [0.6, 0.0, -1],
#                          [-0.6, 0.0, -1],
#                          [0.6, 0.5, -1],
#                          [-0.6, 0.5, -1],
#                          [1.2, 0.5, 0],
#                          [-1.2, 0.5, 0],
#                          [0, 0.5, -1.2],]))

FrontVector = (np.array([
[1.03923, 0.60000, 0.00000],
[0.42426, 0.42426, -1.03923],
[0.00000, 0.60000, -1.03923],
[-0.42426, 0.42426, -1.03923],
[-1.03923, 0.60000, 0.00000],]))

def render_mesh(mesh, resolution=1024, voxel_size=None, index=5, background=None, scale=2, no_fix_normal=True,path=None):
    if voxel_size is None:
        camera_pose = create_pose(FrontVector[index]*scale)
    else:
        camera_pose = create_shapenet_chair_camera_pose(voxel_size=voxel_size)

    render = Render(size=resolution, camera_pose=camera_pose,
                    background=background)

    triangle_id, rendered_image, normal_map, depth_image, p_images = render.render(path=path,
                                                                                   clean=True,
                                                                                   mesh=mesh,
                                                                                   intensity =3.0,
                                                                                   only_render_images=no_fix_normal)
    del render
    
    return rendered_image

def render_point(mesh, resolution=1024, voxel_size=None, index=5, background=None, scale=2, no_fix_normal=True,path=None, num_points=15000):
    if voxel_size is None:
        camera_pose = create_pose(FrontVector[index]*scale)
    else:
        camera_pose = create_shapenet_chair_camera_pose(voxel_size=voxel_size)

    render = Render(size=resolution, camera_pose=camera_pose,
                    background=background)
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # mesh = o3d.io.read_triangle_mesh(str(path))
    # # ポイントクラウドに変換
    # point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

    # point_cloud = np.asarray(point_cloud.points)
    point_cloud = np.load(path)
    # print(point_cloud.shape)
    triangle_id, rendered_image, normal_map, depth_image, p_images = render.render(path=path,
                                                                                   clean=True,
                                                                                   mesh=point_cloud,
                                                                                   intensity = 3.0,
                                                                                   only_render_images=no_fix_normal,
                                                                                   points=True)
    del render
    
    return rendered_image

def render_point_data(mesh,points_np = None, resolution=1024, voxel_size=None, index=5, background=None, scale=2, no_fix_normal=True,path=None, num_points=15000):
    if voxel_size is None:
        camera_pose = create_pose(FrontVector[index]*scale)
    else:
        camera_pose = create_shapenet_chair_camera_pose(voxel_size=voxel_size)

    render = Render(size=resolution, camera_pose=camera_pose,
                    background=background)
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # mesh = o3d.io.read_triangle_mesh(str(path))
    # # ポイントクラウドに変換
    # point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

    # point_cloud = np.asarray(point_cloud.points)
    # point_cloud = np.load(path)
    point_cloud = points_np
    # print(point_cloud.shape)
    triangle_id, rendered_image, normal_map, depth_image, p_images = render.render(path=path,
                                                                                   clean=True,
                                                                                   mesh=point_cloud,
                                                                                   intensity = 3.0,
                                                                                   only_render_images=no_fix_normal,
                                                                                   points=True)
    del render
    
    return rendered_image

if __name__ == "__main__":
    path = "./data/model_normalized copy.obj"
    # trimeshを使ってメッシュを読み込む
    mesh = trimesh.load(path)

    rendered_image = render_mesh(mesh,path=path, resolution=448, voxel_size=None, index=16, background=None, scale=2, no_fix_normal=True)
    rendered_image224 = render_mesh(mesh,path=path, resolution=224, voxel_size=None, index=16, background=None, scale=2, no_fix_normal=True)

    # print(rendered_image.shape)
    ## canny edge detection
    import cv2
    edges = cv2.Canny(rendered_image, 100, 200)
    edges = cv2.bitwise_not(edges)
    edges = cv2.resize(edges, (224, 224))

    edges2 = cv2.Canny(rendered_image224, 100, 200)
    edges2 = cv2.bitwise_not(edges2)

    cv2.imshow('edges448', edges)
    cv2.imshow('edges224', edges2)
    
    cv2.waitKey(0)


    # import matplotlib.pyplot as plt

    # # 画像の描画
    # plt.figure()
    # plt.imshow(rendered_image)
    # plt.axis('off')  # 軸を表示しない
    # plt.savefig('rendered_image1.png')  # 画像をファイルとして保存
    # plt.close()  # プロットを閉じる
