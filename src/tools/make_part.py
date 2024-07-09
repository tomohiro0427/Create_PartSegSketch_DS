from pointnet2_ops import pointnet2_utils
import torch
import torchvision
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()


def shuffule_point_cloud(shape, labels):
    bs = shape.size(0)
    num_points = shape.size(1)
    # 各バッチごとにshuffle_indicesを作成
    shuffle_indices = torch.stack([torch.randperm(num_points) for _ in range(bs)])

    # 各バッチの点群データとラベルをシャッフル
    shuffled_point_cloud_data = shape[torch.arange(bs).unsqueeze(1),shuffle_indices]
    shuffled_labels = labels[torch.arange(bs).unsqueeze(1), shuffle_indices]

    return shuffled_point_cloud_data.contiguous(), shuffled_labels.contiguous()

def point2MakeLabel(shape_fix, shape_edit, num_fixed = 2048, num_edit=1024):
    # サンプリングする数
    sampled_fixed = fps(shape_fix, num_fixed)
    sampled_edit = fps(shape_edit, num_edit)
    shape_fps = torch.cat([sampled_fixed, sampled_edit], dim = 1)

    label_fixed = torch.zeros((sampled_fixed.size(0), sampled_fixed.size(1)))
    label_edit = torch.ones((sampled_edit.size(0), sampled_edit.size(1)))
    labels = torch.cat((label_fixed, label_edit), dim=1)

    shuffled_shape_fps, shffled_labels = shuffule_point_cloud(shape_fps, labels)

    return shuffled_shape_fps, shffled_labels

def get_max_min_points(shape_all):
    shape = shape_all.squeeze(0).cpu().detach().numpy()# [1, N ,3] -> [N, 3]

    # 各次元の最小値と最大値を求める
    min_x, min_y, min_z = np.min(shape, axis=0)
    max_x, max_y, max_z = np.max(shape, axis=0)

    # 8つの角の座標を生成
    corners = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z]
    ])
    return corners



