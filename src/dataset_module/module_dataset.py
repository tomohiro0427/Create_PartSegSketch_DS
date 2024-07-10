import os, logging
import random
import numpy as np
from src.dataset_module import point_operation

import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms
from pathlib import Path
import pandas as pd
import cv2



logger = logging.getLogger(__name__)

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
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

def get_class_id(class_name):
    return cate_to_synsetid[class_name]

def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)

class Uniform15KPC(data.Dataset):
    ''' Uniform15KPC dataset class.
    '''

    def __init__(self, root, subdirs, tr_sample_size=10000, te_sample_size=10000, split='train', scale=1.,
                 normalize_per_shape=False, random_subsample=False, normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None, input_dim=3, standardize_per_shape=False, no_normalize=False):
        self.root = root
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        if split == 'train':
            self.max = tr_sample_size
        elif split == 'val':
            self.max = te_sample_size
        else:
            self.max = max((tr_sample_size, te_sample_size))

        sample_count = 0
        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue

            all_mids = []
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

            # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
            for mid in all_mids:
                obj_fname = os.path.join(root, subd, mid + ".npy")
                try:
                    point_cloud = np.load(obj_fname)  # (15k, 3)
                except:
                    continue

                assert point_cloud.shape[0] == 15000
                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))
                sample_count += 1

        # Shuffle the index deterministically (based on the number of examples)
        # self.all_points = self.all_points[:50]

        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)

        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.all_points, [self.per_points_shift, self.per_points_scale] = point_operation.normalize_point_cloud(self.all_points, verbose=True)

        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d" % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s
        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def get_standardize_stats(self, idx):
        shift = self.per_points_shift[idx].reshape(1, self.input_dim)
        scale = self.per_points_scale[idx].reshape(1, -1)
        return shift, scale    

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def save_statistics(self, save_dir):
        np.save(os.path.join(save_dir, f"{self.split}_set_mean.npy"), self.all_points_mean)
        np.save(os.path.join(save_dir, f"{self.split}_set_std.npy"), self.all_points_std)
        np.save(os.path.join(save_dir, f"{self.split}_set_idx.npy"), np.array(self.shuffle_idx))

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        # tr_out = self.train_points[idx]
        # if self.random_subsample:
        #     tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        # else:
        #     tr_idxs = np.arange(self.tr_sample_size)
        # tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        # te_out = self.test_points[idx]
        # if self.random_subsample:
        #     te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        # else:
        #     te_idxs = np.arange(self.te_sample_size)
        # te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        # tr_ofs = tr_out.mean(0, keepdim=True)
        # te_ofs = te_out.mean(0, keepdim=True)

        shift, scale = self.get_standardize_stats(idx)
        shift, scale = torch.from_numpy(np.asarray(shift)), torch.from_numpy(np.asarray(scale))
        allPoints  = self.all_points[idx]
        
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]
        name = mid.split('/')[-1]

        return {
            'idx': idx,
            # 'pointcloud': tr_out, # if self.split == 'train' else te_out
            # 'pointcloud_ref': te_out,
            'pointcloud_all': allPoints,
            # 'offset': tr_ofs if self.split == 'train' else te_ofs,
            'label': cate_idx,
            'sid': sid, 'mid': mid,
            'name': name,
            'shift': shift, 'scale': scale,
        }

class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(self, 
                 root="/work/vslab2018/3d/data/ShapeNetCore.v2.PC15k/",
                 categories=['airplane'],
                 tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None, standardize_per_shape=True, no_normalize=True):
        self.root = root
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        # assert 'v2' in root, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super().__init__(
            root, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split,
            scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean,
            all_points_std=all_points_std,
            input_dim=3,
            standardize_per_shape=standardize_per_shape, 
            no_normalize=no_normalize)
        


class PointCloudWithSketchDataset(data.Dataset):
    def __init__(self, root_dir, split = 'val', categories = ['chair']):
        if categories not in ['all']:
            synset_ids = [cate_to_synsetid[c] for c in categories]
            self.root_dir = Path(root_dir, synset_ids[0], split)
        else:
            #作り変える
            self.root_dir = Path(root_dir)
        print(self.root_dir)

        self.transform =  transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor()  # これがPIL ImageをTensorに変換します
                                        ])
        self.input_dim = 3
        self.all_points = []
        self.pc_paths = self.find_npy_files()
        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.all_points, [self.per_points_shift, self.per_points_scale] = point_operation.normalize_point_cloud(self.all_points, verbose=True)


        self.datas_dir = self.find_datas_dir()
        

    def find_datas_dir(self):
        dir_data_path = []
        for path in self.root_dir.rglob('shape_all.png'):
            dir_data_path.append(path.parent)
        return dir_data_path
    
    def find_npy_files(self):
        npy_paths = []
        for path_pc in self.root_dir.glob('**/*.npy'):
            npy_paths.append(path_pc.stem)
            point_cloud = np.load(path_pc)
            self.all_points.append(point_cloud[np.newaxis, ...])

        return npy_paths
    
    def get_standardize_stats(self, idx):
        shift = self.per_points_shift[idx].reshape(1, self.input_dim)
        scale = self.per_points_scale[idx].reshape(1, -1)
        return shift, scale  
    
    def image2sketch(self, img_data):
        d = 15
        sigmaColor = 80
        sigmaSpace = 80
        # バイラテラルフィルタの適用
        image = cv2.bilateralFilter(img_data, d, sigmaColor, sigmaSpace)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 200, 500)
        # edges = cv2.bitwise_not(edges)
        # rgb_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # kernel = np.ones((5,5), np.uint8)
        kernel2 = np.ones((3,3), np.uint8)

        test = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel2)
        # test = cv2.dilate(edges, kernel, iterations=1)
        test = cv2.bitwise_not(test)
        rgb_edges = cv2.cvtColor(test, cv2.COLOR_GRAY2RGB)

        return rgb_edges

    def __len__(self):
        return len(self.datas_dir)

    def __getitem__(self, idx):
        img_path = self.datas_dir[idx]

        image_shape_all_path = Path(img_path, 'shape_all.png')
        #ファイルが存在するのか
        if not image_shape_all_path.exists():
            print(f'File not found: {image_shape_all_path}')
        else:
            image_shape_all = cv2.imread(image_shape_all_path)
            image_sketch_all = self.image2sketch(image_shape_all)
            image_sketch_all = self.transform(Image.fromarray(image_sketch_all))

        image_shape_fix_path = Path(img_path, 'shape_fix.png')
        #ファイルが存在するのか
        if not image_shape_fix_path.exists():
            print(f'File not found: {image_shape_fix_path}')
        else:
            image_shape_fix = cv2.imread(image_shape_fix_path)
            image_sketch_fix = self.image2sketch(image_shape_fix)
            image_sketch_fix = self.transform(Image.fromarray(image_sketch_fix))

        image_shape_edit_path = Path(img_path, 'shape_edit.png')
        #ファイルが存在するのか
        if not image_shape_edit_path.exists():
            print(f'File not found: {image_shape_edit_path}')
        else:
            image_shape_edit = cv2.imread(image_shape_edit_path)
            image_sketch_edit = self.image2sketch(image_shape_edit)
            image_sketch_edit = self.transform(Image.fromarray(image_sketch_edit))

        label_path = Path(img_path.parent, f'{img_path.parent.name}_labels_{img_path.parents[1].stem}.parquet')

        #ファイルが存在するのか
        if label_path.exists():
            label_data = pd.read_parquet(label_path)
            # Convert DataFrame to dictionary of numpy arrays
            label_data_dict = {col: torch.tensor(label_data[col].values) for col in label_data.columns}
        else:
            print(f'File not found: {label_path}')

        #point cloudのindexを取得
        index_pc = self.pc_paths.index(img_path.parents[1].stem)
        point_cloud = torch.tensor(self.all_points[index_pc])

        #正規化の値を取得
        shift, scale = self.get_standardize_stats(index_pc)
        shift, scale = torch.from_numpy(np.asarray(shift)), torch.from_numpy(np.asarray(scale))

        
        return {
            'image_all': image_shape_all,
            'image_fix': image_shape_fix,
            'image_edit': image_shape_edit,
            'sketch_all': image_sketch_all,
            'sketch_fix': image_sketch_fix,
            'sketch_edit': image_sketch_edit,
            'label': label_data_dict,
            'point_cloud': point_cloud,
            'path'     : str(img_path),
            'shift': shift, 
            'scale': scale,
        }

class PointCloudWithPartSegLabelDS(data.Dataset):
    def __init__(self, root, split = 'val', categories = ['chair']):
        if categories not in ['all']:
            synset_ids = [cate_to_synsetid[c] for c in categories]
            self.root_dir = Path(root, split, synset_ids[0] )
        else:
            #作り変える
            self.root_dir = Path(root)
        print(self.root_dir)

        self.input_dim = 3
        self.all_points = []
        self.pc_paths = self.find_npy_files()
        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.all_points, [self.per_points_shift, self.per_points_scale] = point_operation.normalize_point_cloud(self.all_points, verbose=True)

    
    def find_npy_files(self):
        npy_paths = []
        for path_pc in self.root_dir.glob('**/*.npy'):
            npy_paths.append(path_pc.stem)
            point_cloud = np.load(path_pc)
            self.all_points.append(point_cloud[np.newaxis, ...])

        return npy_paths
    
    def get_standardize_stats(self, idx):
        shift = self.per_points_shift[idx].reshape(1, self.input_dim)
        scale = self.per_points_scale[idx].reshape(1, -1)
        return shift, scale  


    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, idx):
        pc_name = self.pc_paths[idx]

        label_path = Path(self.root_dir,pc_name, f'{pc_name}.parquet')

        #ファイルが存在するのか
        if label_path.exists():
            label_data = pd.read_parquet(label_path)
            # Convert DataFrame to dictionary of numpy arrays
            label_data_dict = {col: torch.tensor(label_data[col].values) for col in label_data.columns}
        else:
            print(f'File not found: {label_path}')

        #point cloudのindexを取得
        point_cloud = torch.tensor(self.all_points[idx])

        #正規化の値を取得
        shift, scale = self.get_standardize_stats(idx)
        shift, scale = torch.from_numpy(np.asarray(shift)), torch.from_numpy(np.asarray(scale))

        
        return {
            'label': label_data_dict,
            'point_cloud': point_cloud,
            'path'     : str(label_path.parent),
            'shift': shift, 
            'scale': scale,
            'name': pc_name,
        }
    
class PointCloudWithPartSegSketch(data.Dataset):
    def __init__(self, root, split = 'val', categories = ['chair'], get_images = ['edit_sketch']):
        if categories not in ['all']:
            synset_ids = [cate_to_synsetid[c] for c in categories]
            self.root_dir = Path(root, split, synset_ids[0] )
        else:
            #作り変える
            self.root_dir = Path(root)
        print(self.root_dir)

        self.input_dim = 3
        self.all_points = []
        self.all_labels = []
        self.pc_paths = self.find_npy_files()
        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.all_labels = np.concatenate(self.all_labels)  # (N, 15000)
        self.all_points, [self.per_points_shift, self.per_points_scale] = point_operation.normalize_point_cloud(self.all_points, verbose=True)

        self.data_paths = self.find_datas_dir()
        self.get_images = get_images
        for image_name in self.get_images:
            if image_name == 'edit_sketch':
                self.edit_sketch_data = self.get_image_data('edit_sketch.png')
                self.edit_sketch_data = np.concatenate(self.edit_sketch_data)  # (N, 15000, 3)
            elif image_name == 'fix_sketch':
                self.fix_sketch_data = self.get_image_data('fix_sketch.png')
                self.fix_sketch_data = np.concatenate(self.fix_sketch_data)
            elif image_name == 'fix_image':
                self.fix_data = self.get_image_data('fix.png')
                self.fix_data = np.concatenate(self.fix_data)
            elif image_name == 'edit_image':
                self.edit_data = self.get_image_data('edit.png')
                self.edit_data = np.concatenate(self.edit_data)

    
    def find_npy_files(self):
        npy_paths = []
        for path_pc in self.root_dir.glob('**/*.npy'):
            npy_paths.append(path_pc.stem)
            point_cloud = np.load(path_pc)
            self.all_points.append(point_cloud[np.newaxis, ...])

            parquet_data = pd.read_parquet(Path(path_pc.parent, f'{path_pc.stem}.parquet'))
            label_data =  parquet_data['label'].values

            self.all_labels.append(label_data[np.newaxis, ...])

        return npy_paths
    
    def find_datas_dir(self):
        dir_data_path = []
        for path in self.root_dir.rglob('edit_sketch.png'):
            dir_data_path.append(path.parent)
        if len(dir_data_path) == 0:
            print('No edit_sketch.png file found')
        return dir_data_path
    
    def get_image_data(self, name='edit_sketch.png'):
        image_data = []
        for path in self.data_paths:
            image_path = Path(path, name)
            if not image_path.exists():
                print(f'File not found: {image_path}')
            else:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = transforms.Resize((224, 224))(image)
                image = transforms.ToTensor()(image)
                image_data.append(image.unsqueeze(0))
        return image_data
    
    def get_standardize_stats(self, idx):
        shift = self.per_points_shift[idx].reshape(1, self.input_dim)
        scale = self.per_points_scale[idx].reshape(1, -1)
        return shift, scale  


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        
        # 動的にreturn辞書を構築
        return_dict = {}

        for img_name in self.get_images:
            if img_name == 'edit_sketch':
                return_dict['edit_sketch'] = self.edit_sketch_data[idx]
            elif img_name == 'fix_sketch':
                return_dict['fix_sketch'] = self.fix_sketch_data[idx]
            elif img_name == 'fix_image':
                return_dict['fix_image'] = self.fix_data[idx]
            elif img_name == 'edit_image':
                return_dict['edit_image'] = self.edit_data[idx]

        # point cloudのindexを取得
        index_pc = self.pc_paths.index(data_path.parents[1].stem)
        point_cloud = torch.tensor(self.all_points[index_pc])
        label = torch.tensor(self.all_labels[index_pc])

        # 正規化の値を取得
        shift, scale = self.get_standardize_stats(idx)
        shift, scale = torch.from_numpy(np.asarray(shift)), torch.from_numpy(np.asarray(scale))

        # return辞書に他の固定的な値を追加
        return_dict.update({
            'point_cloud': point_cloud,
            'label': label,
            'path': str(data_path),
            'shift': shift,
            'scale': scale,
            'name': data_path.parents[1].stem,
        })
        
        return return_dict