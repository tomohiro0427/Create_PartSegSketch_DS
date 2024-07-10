import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
import sys
import cv2
import torch
import gc
from pathlib import Path

'''
## 改良点##
1. yamlファイルでディレクトリや数を指定
2. front back right left side のカメラ指定
3. ズーム距離指定
'''
def center_crop(image, size=224):
    # 画像の幅と高さを取得
    height, width = image.shape[:2]

    # クロップするサイズを決定
    crop_size = min(width, height)//2  # 幅と高さのうち小さい方を基準にクロップする

    # 中心座標を計算
    center_x, center_y = width // 2, height // 2

    # クロップする領域の左上の座標を計算
    x_start = center_x - crop_size // 2
    y_start = center_y - crop_size // 2

    # クロップする領域の幅と高さを指定
    crop_width, crop_height = crop_size, crop_size

    # 指定した領域で画像をクロップ
    cropped_image = image[y_start:y_start+crop_height, x_start:x_start+crop_width]
    cropped_image = cv2.resize(cropped_image, (size,size))

    return cropped_image

def point2image(point_cloud_data, view_params, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if type(point_cloud_data) == torch.Tensor:
        point_cloud_data = point_cloud_data.cpu().detach().numpy()

    img_path = []

    for i in range(point_cloud_data.shape[0]):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data[i])
        # print(point_cloud_data[i].shape)

        # カメラのビューパラメータ
        view_params = {

            "zoom": 2.5,
            "front": [-1.0, 1.0, -1.0],
            "lookat": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0]
        }

        # Visualizer を作成し、ウィンドウを非表示で開く
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        # vis.create_window(visible=False, width=224, height=224)
        vis.add_geometry(point_cloud)

        # カメラのビューパラメータを設定
        ctr = vis.get_view_control()
        ctr.set_zoom(view_params["zoom"])
        ctr.set_front(view_params["front"])
        ctr.set_lookat(view_params["lookat"])
        ctr.set_up(view_params["up"])
        ctr.change_field_of_view(step=5)

        # # 点のサイズを設定
        # render_option = vis.get_render_option()
        # render_option.point_size = 1.6  # 点のサイズを大きく設定

        # ジオメトリの更新とレンダラーの更新
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        save_path_img = os.path.join(save_path, 'points_img')
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)

        # 画像をキャプチャして保存し、ウィンドウを破棄
        img_name = f'{save_path_img}/{i}.png'
        vis.capture_screen_image(img_name,do_render=True)
        vis.destroy_window()

        img_path.append(img_name)


        # 不要な変数を削除
        del ctr
        del vis
        gc.collect()


    return img_path

def image2sketch(img_path, save_path):
    sketches = []
    for one_img_path in img_path:
        # 画像を読み込む
        img = cv2.imread(one_img_path)

        cropped_image = center_crop(img, size=224)

        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # Cannyエッジ検出を適用
        edges = cv2.Canny(gray, 200, 500)
        edges = cv2.bitwise_not(edges)
        # print(edges.shape)

        save_path_sketch = os.path.join(save_path, 'sketch')
        if not os.path.exists(save_path_sketch):
            os.makedirs(save_path_sketch)
        save_path_gray = os.path.join(save_path, 'gray')
        if not os.path.exists(save_path_gray):
            os.makedirs(save_path_gray)
        

        img_name = one_img_path.split('/')[-1]
    
        # # 保存
        cv2.imwrite(os.path.join(save_path_sketch, img_name), edges)
        cv2.imwrite(os.path.join(save_path_gray, img_name), gray)

        rgb_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        sketches.append(rgb_edges)
    
    sketches = torch.tensor(sketches)

    return sketches

def pointWithLabel2image(point_cloud_data, labels, view_params, save_path):

    # save_path = os.path.join(save_path, 'pointWithLabel')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if type(point_cloud_data) == torch.Tensor:
        point_cloud_data = point_cloud_data.cpu().detach().numpy()
    if type(labels) == torch.Tensor:
         labels = labels.cpu().detach().numpy()

    img_path = []

    for i in range(point_cloud_data.shape[0]):

        # ラベルに基づいて赤と青の色を割り当てる
        colors = np.zeros((point_cloud_data.shape[1], 3))  # 初期化：すべての点を黒色で初期化
        colors[labels[i] == 0] = [1, 0, 0]  # ラベル0の点を赤色に設定
        colors[labels[i] == 1] = [0, 0, 1]  # ラベル1の点を青色に設定

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data[i])
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        # print(point_cloud_data[i].shape)

        # カメラのビューパラメータ
        view_params = {

            "zoom": 2.5,
            "front": [-1.0, 1.0, -1.0],
            "lookat": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0]
        }

        # Visualizer を作成し、ウィンドウを非表示で開く
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        # vis.create_window(visible=False, width=224, height=224)
        vis.add_geometry(point_cloud)

        # カメラのビューパラメータを設定
        ctr = vis.get_view_control()
        ctr.set_zoom(view_params["zoom"])
        ctr.set_front(view_params["front"])
        ctr.set_lookat(view_params["lookat"])
        ctr.set_up(view_params["up"])
        ctr.change_field_of_view(step=5)

        # # 点のサイズを設定
        # render_option = vis.get_render_option()
        # render_option.point_size = 1.6  # 点のサイズを大きく設定

        # ジオメトリの更新とレンダラーの更新
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        # 画像をキャプチャして保存し、ウィンドウを破棄
        img_name = f'{save_path}/{i}.png'
        vis.capture_screen_image(img_name,do_render=True)
        vis.destroy_window()

        img_path.append(img_name)


        # 不要な変数を削除
        del ctr
        del vis


    return 


def point2imageWithCorner(point_cloud_data, corners, save_path, name ):
    
    if type(point_cloud_data) == torch.Tensor:
        point_cloud_data = point_cloud_data.squeeze(0).cpu().detach().numpy()
    if type(corners) == torch.Tensor:
        corners = corners.squeeze(0).cpu().detach().numpy()


    pc_corner = o3d.geometry.PointCloud()
    colors = np.tile([1, 1, 1], (8, 1))  # 白色にする
    pc_corner.points = o3d.utility.Vector3dVector(corners)
    pc_corner.colors = o3d.utility.Vector3dVector(colors)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
    # print(point_cloud_data[i].shape)

    # カメラのビューパラメータ
    view_params = {

        "zoom": 2.5,
        "front": [-1.0, 1.0, -1.0],
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
        }

    # Visualizer を作成し、ウィンドウを非表示で開く
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    # vis.create_window(visible=False, width=224, height=224)
    vis.add_geometry(pc_corner)
    vis.add_geometry(point_cloud)
    

    # カメラのビューパラメータを設定
    ctr = vis.get_view_control()
    ctr.set_zoom(view_params["zoom"])
    ctr.set_front(view_params["front"])
    ctr.set_lookat(view_params["lookat"])
    ctr.set_up(view_params["up"])
    ctr.change_field_of_view(step=5)

    # # 点のサイズを設定
    # render_option = vis.get_render_option()
    # render_option.point_size = 1.6  # 点のサイズを大きく設定

    # ジオメトリの更新とレンダラーの更新
    vis.update_geometry(pc_corner)
    vis.update_geometry(point_cloud)
    

    vis.poll_events()
    vis.update_renderer()

    save_path_img = os.path.join(save_path, 'points_img')
    if not os.path.exists(save_path_img):
        os.makedirs(save_path_img)

    # 画像をキャプチャして保存し、ウィンドウを破棄
    img_name = f'{save_path_img}/{name}.png'
    vis.capture_screen_image(img_name,do_render=True)
    vis.destroy_window()


    # 不要な変数を削除
    del ctr
    del vis


    return img_name

def OnePointWithLabel2image(point_cloud_data, labels,  corners, view_params=None, save_path=None):

    # save_path = os.path.join(save_path, 'pointWithLabel')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if type(point_cloud_data) == torch.Tensor:
        point_cloud_data = point_cloud_data.cpu().detach().numpy()
    if type(labels) == torch.Tensor:
         labels = labels.cpu().detach().numpy()

    point_cloud_data = point_cloud_data.squeeze(0)
    labels = labels.squeeze(0)

    part = np.unique(labels)

    pc_corner = o3d.geometry.PointCloud()
    colors = np.tile([1, 1, 1], (8, 1))  # 白色にする
    pc_corner.points = o3d.utility.Vector3dVector(corners)
    pc_corner.colors = o3d.utility.Vector3dVector(colors)

    cl = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],])


    img_path = []

    # ラベルに基づいて赤と青の色を割り当てる
    colors = np.zeros((point_cloud_data.shape[0], 3))  # 初期化：すべての点を黒色で初期化
    for j, label in enumerate(part):
        colors[labels== label] = cl[j]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    # print(point_cloud_data[i].shape)

    # カメラのビューパラメータ
    view_params = {

        "zoom": 1.5,
        "front": [-1.0, 1.0, -1.0],
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    }

    # Visualizer を作成し、ウィンドウを非表示で開く
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    # vis.create_window(visible=False, width=224, height=224)
    vis.add_geometry(point_cloud)
    vis.add_geometry(pc_corner)

    # カメラのビューパラメータを設定
    ctr = vis.get_view_control()
    ctr.set_zoom(view_params["zoom"])
    ctr.set_front(view_params["front"])
    ctr.set_lookat(view_params["lookat"])
    ctr.set_up(view_params["up"])
    ctr.change_field_of_view(step=5)
    vis.update_geometry(pc_corner)

    # # 点のサイズを設定
    # render_option = vis.get_render_option()
    # render_option.point_size = 1.6  # 点のサイズを大きく設定

    # ジオメトリの更新とレンダラーの更新
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()

    # 画像をキャプチャして保存し、ウィンドウを破棄
    img_name = f'{save_path}/syvete.png'
    vis.capture_screen_image(img_name,do_render=True)
    vis.destroy_window()

    img_path.append(img_name)


    # 不要な変数を削除
    del ctr
    del vis


    return 

def point2imageWithLabelView(point_cloud_data, label, corners, view='X-1Y1Z-1', editOrFix='edit', save_path=None):


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if type(point_cloud_data) == torch.Tensor:
        point_cloud_data = point_cloud_data.cpu().detach().numpy()
    if type(label) == torch.Tensor:
         label = label.cpu().detach().numpy()
    if type(corners) == torch.Tensor:
        corners = corners.squeeze(0).cpu().detach().numpy()

    point_cloud_data = point_cloud_data.squeeze(0)
    label = label.squeeze(0)

    part = np.unique(label)

    # 8つの角の座標を生成
    pc_corner = o3d.geometry.PointCloud()
    colors = np.tile([1, 1, 1], (8, 1))  # 白色にする
    pc_corner.points = o3d.utility.Vector3dVector(corners)
    pc_corner.colors = o3d.utility.Vector3dVector(colors)

    view_dict = {
                'X-1Y1Z-1': [-1.0, 1.0, -1.0],
                'X-0.5Y1Z-1': [-0.5, 1.0, -1.0],
                'X0Y1Z-1': [0.0, 1.0, -1.0],
                'X0.5Y1Z-1': [0.5, 1.0, -1.0],
                'X1Y1Z-1': [1.0, 1.0, -1.0],
                'X-1Y1Z-0.5': [-1.0, 1.0, -0.5],
                'X1Y1Z-0.5': [1.0, 1.0, -0.5],
                }



    # カメラのビューパラメータ
    view_params = {

        "zoom": 2.5,
        "front": view_dict[view],
        #  "front": [-1.0, 1.0, -1.0],
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    }
    save_path_view = os.path.join(save_path, view)
    if not os.path.exists(save_path_view):
        os.makedirs(save_path_view)

    img_path = []
    
    for i in part:

        mask = label == i
        if editOrFix == 'fix':#複数なら反転
            mask = ~mask
        filtered_points = point_cloud_data[mask]
        
        # 100点未満の場合は誤ラベルの可能性があるためスキップ
        if point_cloud_data[mask].shape[0] < 100 or point_cloud_data[~mask].shape[0] < 100:
            continue

        

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        # point_cloud.colors = o3d.utility.Vector3dVector(colors)
        # print(point_cloud_data[i].shape)


        # Visualizer を作成し、ウィンドウを非表示で開く
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        # vis.create_window(visible=False, width=224, height=224)
        vis.add_geometry(point_cloud)
        vis.add_geometry(pc_corner)

        # カメラのビューパラメータを設定
        ctr = vis.get_view_control()
        ctr.set_zoom(view_params["zoom"])
        ctr.set_front(view_params["front"])
        ctr.set_lookat(view_params["lookat"])
        ctr.set_up(view_params["up"])
        ctr.change_field_of_view(step=5)
        vis.update_geometry(pc_corner)

        # # 点のサイズを設定
        # render_option = vis.get_render_option()
        # render_option.point_size = 1.6  # 点のサイズを大きく設定

        # ジオメトリの更新とレンダラーの更新
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        # save_path_part = Path(save_path) / f'view[{view}]' / f'label[{i}]'
        save_path_part = os.path.join(save_path_view, f'{i}')
        if not os.path.exists(save_path_part):
            os.makedirs(save_path_part)

        # 画像をキャプチャして保存し、ウィンドウを破棄
        # img_name = save_path_part/ f'{editOrFix}.png'
        img_name = f'{save_path_part}/{editOrFix}.png'
        vis.capture_screen_image(img_name,do_render=True)
        vis.destroy_window()

        img_path.append(img_name)


        # 不要な変数を削除
        del ctr
        del vis


    return img_path


def point2imageWithView(point_cloud_data, corners, view='X-1Y1Z-1', save_path=None, name='point_all'):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if type(point_cloud_data) == torch.Tensor:
        point_cloud_data = point_cloud_data.cpu().detach().numpy()
    if type(corners) == torch.Tensor:
        corners = corners.squeeze(0).cpu().detach().numpy()

    point_cloud_data = point_cloud_data.squeeze(0)


    # 8つの角の座標を生成
    pc_corner = o3d.geometry.PointCloud()
    colors = np.tile([1, 1, 1], (8, 1))  # 白色にする
    pc_corner.points = o3d.utility.Vector3dVector(corners)
    pc_corner.colors = o3d.utility.Vector3dVector(colors)

    view_dict = {
                'X-1Y1Z-1': [-1.0, 1.0, -1.0],
                'X-0.5Y1Z-1': [-0.5, 1.0, -1.0],
                'X0Y1Z-1': [0.0, 1.0, -1.0],
                'X0.5Y1Z-1': [0.5, 1.0, -1.0],
                'X1Y1Z-1': [1.0, 1.0, -1.0],
                'X-1Y1Z-0.5': [-1.0, 1.0, -0.5],
                'X1Y1Z-0.5': [1.0, 1.0, -0.5],
                }

    # カメラのビューパラメータ
    view_params = {

        "zoom": 2.5,
        "front": view_dict[view],
        #  "front": [-1.0, 1.0, -1.0],
        "lookat": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    }
    save_path_view = os.path.join(save_path, f'{view}')
    if not os.path.exists(save_path_view):
        os.makedirs(save_path_view)

    img_path = []

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)


    # Visualizer を作成し、ウィンドウを非表示で開く
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    # vis.create_window(visible=False, width=224, height=224)
    vis.add_geometry(point_cloud)
    vis.add_geometry(pc_corner)

    # カメラのビューパラメータを設定
    ctr = vis.get_view_control()
    ctr.set_zoom(view_params["zoom"])
    ctr.set_front(view_params["front"])
    ctr.set_lookat(view_params["lookat"])
    ctr.set_up(view_params["up"])
    ctr.change_field_of_view(step=5)
    vis.update_geometry(pc_corner)

    # # 点のサイズを設定
    # render_option = vis.get_render_option()
    # render_option.point_size = 1.6  # 点のサイズを大きく設定

    # ジオメトリの更新とレンダラーの更新
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()

    # 画像をキャプチャして保存し、ウィンドウを破棄
    # img_name = save_path_part/ f'{editOrFix}.png'
    img_name = f'{save_path_view}/{name}.png'
    vis.capture_screen_image(img_name,do_render=True)
    vis.destroy_window()

    img_path.append(img_name)


    # 不要な変数を削除
    del ctr
    del vis


    return img_path

def rmoveOnePointWithLabel2image(point_cloud_data, labels,  corners, view_params=None, save_path=None):

    # save_path = os.path.join(save_path, 'pointWithLabel')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if type(point_cloud_data) == torch.Tensor:
        point_cloud_data = point_cloud_data.cpu().detach().numpy()
    if type(labels) == torch.Tensor:
         labels = labels.cpu().detach().numpy()

    point_cloud_data = point_cloud_data.squeeze(0)
    labels = labels.squeeze(0)

    part = np.unique(labels)

    pc_corner = o3d.geometry.PointCloud()
    colors = np.tile([1, 1, 1], (8, 1))  # 白色にする
    pc_corner.points = o3d.utility.Vector3dVector(corners)
    pc_corner.colors = o3d.utility.Vector3dVector(colors)

    cl = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],])

    for i in part:

        mask = labels == i
        mask = ~mask
        filtered_points = point_cloud_data[mask]
        filtered_labels = labels[mask]

        img_path = []

        # ラベルに基づいて赤と青の色を割り当てる
        colors = np.zeros((filtered_points.shape[0], 3))  # 初期化：すべての点を黒色で初期化
        for j, label in enumerate(part):
            colors[filtered_labels== label] = cl[j]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        # print(point_cloud_data[i].shape)

        # カメラのビューパラメータ
        view_params = {

            "zoom": 1.5,
            "front": [-1.0, 1.0, -1.0],
            "lookat": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0]
        }

        # Visualizer を作成し、ウィンドウを非表示で開く
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        # vis.create_window(visible=False, width=224, height=224)
        vis.add_geometry(point_cloud)
        vis.add_geometry(pc_corner)

        # カメラのビューパラメータを設定
        ctr = vis.get_view_control()
        ctr.set_zoom(view_params["zoom"])
        ctr.set_front(view_params["front"])
        ctr.set_lookat(view_params["lookat"])
        ctr.set_up(view_params["up"])
        ctr.change_field_of_view(step=5)
        vis.update_geometry(pc_corner)

        # # 点のサイズを設定
        # render_option = vis.get_render_option()
        # render_option.point_size = 1.6  # 点のサイズを大きく設定

        # ジオメトリの更新とレンダラーの更新
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        # 画像をキャプチャして保存し、ウィンドウを破棄
        img_name = f'{save_path}/{i}.png'
        vis.capture_screen_image(img_name,do_render=True)
        vis.destroy_window()

        img_path.append(img_name)


        # 不要な変数を削除
        del ctr
        del vis


    return 

