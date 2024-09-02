from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
# from data import ShapeNetPart
from model import DGCNN_partseg
import numpy as np
from torch.utils.data import DataLoader
# from util import cal_loss, IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement
from src.dataset_module import module_dataset
from src.tools.make_part import fps, get_max_min_points, get_upsampled_labels
from src.sketch.point2image import point2image, image2sketch, pointWithLabel2image,rmoveOnePointWithLabel2image, point2imageWithCorner,OnePointWithLabel2image
from src.dataset_module.point_operation import rotate_point_cloud_by_angle_batch
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import shutil


class_choices = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    'all': 'all'
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


def test(args):
    # test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice),
    #                          batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    train_dataset = module_dataset.ShapeNet15kPointClouds(
                                                    # root = "/home/tomohiro/VAL/Train_CanonicalVAE/mini_dataset_3",
                                                    # root = "/home/tomohiro/workspace/Dataset/ShapeNetCore.v2.PC15k/",
                                                    root = args.dataset_path,
                                                    split = args.split,
                                                    categories=[args.categories], 
                                                    )
    train_Dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    # #Try to load models
    # seg_num_all = test_loader.dataset.seg_num_all
    seg_num_all = 50
    # seg_start_index = test_loader.dataset.seg_start_index
    # partseg_colors = test_loader.dataset.partseg_colors
    if args.model == 'dgcnn':
        model = DGCNN_partseg(args, seg_num_all).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    print(args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    #arg.output_pathのフォルダを作成
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # splitとoutput_pathを結合
    output_path = Path(args.output_path)
    outputSplit_path = output_path / args.split
    if not os.path.exists(outputSplit_path):
        os.makedirs(outputSplit_path)
    
    # outputpaht/val/chair_id/
    outputCategory_path = outputSplit_path / cate_to_synsetid[args.categories]
    if not os.path.exists(outputCategory_path):
        os.makedirs(outputCategory_path)


    with tqdm(train_Dataloader) as tqdmbar:
        for i, batch in enumerate(tqdmbar):
            
            name = batch.get('name')
            outputName_path = outputCategory_path / name[0] # outputpaht/val/chair_id/name
            if not os.path.exists(outputName_path):
                os.makedirs(outputName_path)

            # print(batch)
            shape_all = batch.get('pointcloud_all').float().cuda()
            data = fps(shape_all, 2048)
            data = rotate_point_cloud_by_angle_batch(data.cpu(), np.pi/2)
            data = torch.from_numpy(data).float().cuda()
            data = data.permute(0, 2, 1)
            # label = batch.get('label').cuda()
            label = torch.tensor([class_choices.index(args.categories)]).cuda()

            label_one_hot = torch.nn.functional.one_hot(label, num_classes=16).float().cuda()
            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            pred_labels = seg_pred.max(dim=2)[1]
            data = rotate_point_cloud_by_angle_batch(data.permute(0, 2, 1).cpu(), -np.pi/2)
            data = torch.from_numpy(data).float().cuda()
            # pointWithLabel2image(data.permute(0, 2, 1), pred, save_path= "./test_out")
            #時間計測
            
            start = time.time()
            labels_shape_all= get_upsampled_labels(shape_all, data, pred_labels)
            tm = time.time() - start
            #shape_allをnpyファイルで保存
            # np.save(outputName_path / f"{name[0]}.npy", shape_all.squeeze(0).cpu().detach().numpy())
            #特定のnpyファイルをコピー
            src_path = Path(args.dataset_path) / cate_to_synsetid[args.categories] / args.split / f"{name[0]}.npy"
            #ファイルが存在するか
            if not os.path.exists(src_path):
                print(f"{src_path} is not found")
                break
            else:
                shutil.copy(src_path, outputName_path)
            #labels_shape_allをpandasのDataFrameに変換にして、parquetで保存
            labels_shape_all = labels_shape_all.squeeze(0).cpu().detach().numpy()
            # print(labels_shape_all.shape)
            df = pd.DataFrame({
                    'label': labels_shape_all
                })
            csv_path = outputName_path / f"{name[0]}.parquet"
            df.to_parquet(csv_path)


           

            #保存ファイルの作成
            #label保存
            #npyファイルのコピー


            # point2image(uppart, save_path= "./test_out")

            # corners = get_max_min_points(data)
            # rmoveOnePointWithLabel2image(data, pred_labels,corners, save_path= "./test_out/labels")
            # rmoveOnePointWithLabel2image(shape_all, labels_shape_all,corners, save_path= "./test_out/labels2")
            # OnePointWithLabel2image(shape_all, labels_shape_all,corners, save_path= "./test_out/labels0")

            # part_idx = np.unique(labels_shape_all.detach().cpu().numpy())
            # for i in part_idx:
            #     mask = labels_shape_all == i
            #     mask = ~mask
            #     filtered_points = shape_all[mask]
            #     print(filtered_points.shape)
            #     _ = point2imageWithCorner(filtered_points, corners, save_path= "./test_out/labels3", name = str(i))
            tqdmbar.set_postfix( crop_time=tm, name=name)
            if i == 2:
                break
            # break


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, 
                        default='pretrained/model.partseg.t7', 
                        # default='pretrained/model.partseg.airplane.t7', 
                        
                        metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    parser.add_argument('--dataset_path', type=str, 
                        default='../Dataset/ShapeNetCore.v2.PC15k/',
                        help='file format of visualization')
    parser.add_argument('--split', type=str, default='val',
                        help='file format of visualization')
    parser.add_argument('--categories', type=str, default='chair',
                        help='file format of visualization')
    parser.add_argument('--output_path', type=str, default='./output/',
                        help='file format of visualization')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    test(args)
