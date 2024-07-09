
import pyrender
import trimesh
import numpy as np
import pyrr
import cv2
import torch
import os

def create_pose(eye):
    target = np.zeros(3)
    camera_pose = np.array(pyrr.Matrix44.look_at(eye=eye,
                                                 target=target,
                                                 up=np.array([0.0, 1.0, 0])).T)
    return np.linalg.inv(np.array(camera_pose))

def PointCloud2Image(point_cloud):
    if type(point_cloud) == torch.Tensor:
        point_cloud = point_cloud.cpu().detach().numpy()

    # 点群データをTrimeshのPointCloudオブジェクトに変換
    cloud = trimesh.points.PointCloud(point_cloud)

    # PyrenderのMeshオブジェクトに変換
    mesh = pyrender.Mesh.from_points(cloud.vertices)
    # シーンの作成とメッシュの追加
    scene = pyrender.Scene()
    scene.add(mesh)

    # カメラの設定
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)
    eye = [0.7,0.7,-0.7]

    camera_pose = create_pose(eye)
    scene.add(camera, pose=camera_pose)

    # ライトの設定
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    scene.add(light, pose=camera_pose)

    # レンダリング
    r = pyrender.OffscreenRenderer(viewport_width=224, viewport_height=224)
    color, depth = r.render(scene)

    return color, depth

def PointCloud2ImageBatch(point_cloud, save = False, save_path = None):
    if type(point_cloud) == torch.Tensor:
            point_cloud = point_cloud.cpu().detach().numpy()

    color_images = []
    depth_images = []
    for i in range(point_cloud.shape[0]):

        # 点群データをTrimeshのPointCloudオブジェクトに変換
        cloud = trimesh.points.PointCloud(point_cloud[i])

        # PyrenderのMeshオブジェクトに変換
        mesh = pyrender.Mesh.from_points(cloud.vertices)
        # シーンの作成とメッシュの追加
        scene = pyrender.Scene()
        scene.add(mesh)

        # カメラの設定
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)
        eye = [0.7,0.7,-0.7]

        camera_pose = create_pose(eye)
        scene.add(camera, pose=camera_pose)

        # ライトの設定
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        scene.add(light, pose=camera_pose)

        # レンダリング
        r = pyrender.OffscreenRenderer(viewport_width=224, viewport_height=224)
        color, depth = r.render(scene)

        if save:
            color_path = os.path.join(save_path,"color")
            depth_path = os.path.join(save_path,"depth")
            if not os.path.exists(color_path):
                os.makedirs(color_path)
            if not os.path.exists(depth_path):
                os.makedirs(depth_path)

            cv2.imwrite(os.path.join(color_path, f"color_{i}.png"), color)
            cv2.imwrite(os.path.join(depth_path, f"depth_{i}.png"), depth)
        
        uint8_depth = (depth * 255).astype(np.uint8)
        
        rgb_depth = cv2.cvtColor(uint8_depth, cv2.COLOR_GRAY2RGB)

        color_images.append(color)
        depth_images.append(rgb_depth)

    color_images= torch.tensor(color_images)
    depth_images= torch.tensor(depth_images)

    return color_images, depth_images


if __name__ == '__main__':
    # npyファイルから点群データを読み込む
    npy_file = './output/chair.npy'  # npyファイルのパス
    point_cloud = np.load(npy_file)

    color, depth = PointCloud2Image(point_cloud)
    print(type(depth))
    print(depth.dtype)
    print(color.shape, depth.shape)

    depth_image = depth

    # Depth画像の正規化（0から1の範囲に変換）
    normalized_depth = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))

    # 0から255の範囲にスケーリングし、8ビットの符号なし整数型に変換
    scaled_depth = (normalized_depth * 255).astype(np.uint8)

    # CLAHEを作成
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # CLAHEを適用
    clahe_image = clahe.apply(scaled_depth)

    edges = cv2.Canny(clahe_image, 200, 500)
    edges = cv2.bitwise_not(edges)

    # 表示して確認
    cv2.imshow('Original Depth', scaled_depth)
    cv2.imshow('CLAHE Image', clahe_image)
    cv2.imshow('Edges', edges)
    cv2.imshow('Color', color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
