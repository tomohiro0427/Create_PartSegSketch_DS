from src.dataset_module import module_dataset
from torch.utils.data import DataLoader
from src.tools.make_part import get_max_min_points
from src.sketch.point2image import point2imageWithLabelView,point2imageWithView
from src.sketch.image2sketch import centerCrop_images,image2sketch
import os
import argparse
from tqdm import tqdm
import numpy as np
from src.render import pyrender_util
import cv2
import torch
from src.unet import UNet
import pandas as pd


def predict_sketches(unet, render_images):
    render_images = torch.from_numpy(render_images/ 255).float().permute(2, 0, 1).unsqueeze(0)
    unet = unet.cuda()
    render_images = render_images.cuda()
    sketchs_all_points = unet(render_images)
    pred_sketchs = sketchs_all_points.squeeze(0).cpu()
    pred = torch.clamp(pred_sketchs, max=1.0).detach().numpy()
    threshold = 0.55
    thresh = np.where(np.mean(pred, axis=0) > threshold, 1.0, 0.0)
    pred = np.stack([thresh]*3, axis=0).transpose(1, 2, 0) * 255
    pred = pred.astype(np.uint8)
    return pred


def CreatePoint2Sketch(args):
    dataset = module_dataset.PointCloudWithPartSegLabelDS(
                                                    root = args.dataset_path,
                                                    split = args.split,
                                                    categories = [args.class_choice],)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    unet = UNet(n_channels=3, n_classes=3, bilinear=False)
    unet.load_state_dict(torch.load('./src/pretrained/exp_best.pth'))
    unet.eval()

    with tqdm(dataloader) as tqdmbar:
        for i, batch in enumerate(tqdmbar):
            label = batch.get('label')['label']# [B, N]
            points = batch.get('point_cloud')# [B, N, 3]
            name = batch.get('name')[0]

            class_id = module_dataset.get_class_id(args.class_choice)

            if args.save_TF:
                data_path = os.path.join(args.save_path,args.split, class_id,name)
            else:
                data_path = os.path.join(args.dataset_path,args.split, class_id,name)
            if not os.path.exists(data_path):
                os.makedirs(data_path)


            ### pyrenderによる画像変換 ### 
            # uniqueでラベルの数を取得
            label = label.squeeze(0).cpu().numpy()
            point_cloud_data = points.squeeze(0).cpu().numpy()
            corners = get_max_min_points(points)#点群の最大値と最小値から8つの角の座標を取得
            part = np.unique(label)

            for part_label in tqdm(part, desc=f"part", leave=False):
                # 部位ごとのラベルで分割
                mask = label == part_label
                
                # 100点未満の場合は誤ラベルの可能性があるためスキップ
                if point_cloud_data[mask].shape[0] < 100 or point_cloud_data[~mask].shape[0] < 100:
                    continue

                # edit部分
                edit_labels = mask
                fix_labels = ~mask
                edit_filtered_points = point_cloud_data[edit_labels]
                fix_filtered_points = point_cloud_data[fix_labels]
                edit_points = np.concatenate([edit_filtered_points, corners], axis=0)
                fix_points = np.concatenate([fix_filtered_points, corners], axis=0)

                # pyrenderによる15視点での画像に変換
                for view_index in tqdm(range(args.num_view), desc=f"View ", leave=False):
                    view_path = os.path.join(data_path, f"{view_index}")
                    if not os.path.exists(view_path):
                        os.makedirs(view_path)
                    partial_path = os.path.join(view_path, f"{part_label}")
                    if not os.path.exists(partial_path):
                        os.makedirs(partial_path)
                    # すべての点群を画像に変換
                    if not os.path.exists(os.path.join(view_path , "shape_all.png")):
                        all_points_data = np.concatenate([point_cloud_data, corners], axis=0)
                        all_points_png = pyrender_util.render_point_data(mesh=None, points_np=all_points_data, resolution=224, voxel_size=None, index=view_index, background=None, scale=2, no_fix_normal=True)
                        all_points_path = os.path.join(view_path, 'all_points.png')
                        cv2.imwrite(all_points_path, all_points_png)
                        # sketch
                        sketch_all = predict_sketches(unet, all_points_png)
                        #スケッチの保存
                        sketch_all_path = os.path.join(view_path, 'all_sketch.png')
                        cv2.imwrite(sketch_all_path, sketch_all)

                    # pyrenderによる画像変換
                    edit_points_png = pyrender_util.render_point_data(mesh=None, points_np=edit_points, resolution=224, voxel_size=None, index=view_index, background=None, scale=2, no_fix_normal=True)
                    fix_points_png = pyrender_util.render_point_data(mesh=None, points_np=fix_points, resolution=224, voxel_size=None, index=view_index, background=None, scale=2, no_fix_normal=True)
                    # 画像の保存
                    fix_points_path = os.path.join(partial_path, 'fix_points.png')
                    edit_points_path = os.path.join(partial_path, 'edit_points.png')
                    cv2.imwrite(fix_points_path, fix_points_png)
                    cv2.imwrite(edit_points_path, edit_points_png)
                    
                    # unetによるスケッチ変換
                    sketch_fix = predict_sketches(unet, fix_points_png)
                    sketch_edit = predict_sketches(unet, edit_points_png)
                    #スケッチの保存
                    sketch_fix_path = os.path.join(partial_path, 'fix_sketch.png')
                    sketch_edit_path = os.path.join(partial_path, 'edit_sketch.png')
                    cv2.imwrite(sketch_fix_path, sketch_fix)
                    cv2.imwrite(sketch_edit_path, sketch_edit)

                    # label の保存　
                    parquet_labels = np.zeros(fix_labels.shape)  # B N
                    parquet_labels[fix_labels == True] = 1 #  B N

                    # num_ones = np.count_nonzero(fix_labels == True)
                    # print(f"Number of ones: {num_ones}")
                    # print(fix_filtered_points.shape)

                    # numpy配列をpandasのDataFrameに変換
                    df = pd.DataFrame({
                        'label': parquet_labels,
                        'fix_bool': fix_labels,
                        'edit_bool': edit_labels,
                    })
                    csv_path = os.path.join(partial_path, f'{part_label}_{name}.parquet')
                    df.to_parquet(csv_path)


            tqdmbar.set_postfix(name=name)

            # break   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create PartSeg Sketch Pyrender')
    parser.add_argument('--exp_name', type=str, default='pyrender Part Segmentation Dataset ', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--class_choice', type=str, default='chair', metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataset_path', type=str, 
                        default='./output/',
                        help='file format of visualization')
    parser.add_argument('--split', type=str, default='val', metavar='N',
                        choices=['train', 'val'])
    parser.add_argument('--editOrFix', type=str, default='fix', metavar='N',
                        choices=['fix', 'edit'])
    parser.add_argument('--save_path', type=str, default='./output/', metavar='N',
                        help='file format of visualization')
    parser.add_argument('--save_TF', action='store_true', default=False,)  
    parser.add_argument('--num_view', type=int, default=14, metavar='N',)
    args = parser.parse_args()

    CreatePoint2Sketch(args)