from src.dataset_module import module_dataset
from torch.utils.data import DataLoader
from src.tools.make_part import get_max_min_points
from src.sketch.point2image import point2imageWithLabelView,point2imageWithView
from src.sketch.image2sketch import centerCrop_images,image2sketch
import os
import argparse
from tqdm import tqdm

def CreatePoint2Sketch(args):
    dataset = module_dataset.PointCloudWithPartSegLabelDS(
                                                    root = args.dataset_path,
                                                    split = args.split,
                                                    categories = [args.class_choice],)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with tqdm(dataloader) as tqdmbar:
        for i, batch in enumerate(tqdmbar):
            label = batch.get('label')['label']# [B, N]
            points = batch.get('point_cloud')# [B, N, 3]
            name = batch.get('name')[0]

            class_id = module_dataset.get_class_id(args.class_choice)

            view = args.view
            if args.save_TF:
                save_path = os.path.join(args.save_path,args.split, class_id,name)
            else:
                save_path = os.path.join(args.dataset_path,args.split, class_id,name)

            editOrFix = args.editOrFix

            corners = get_max_min_points(points)#点群の最大値と最小値から8つの角の座標を取得
            img_paths = point2imageWithLabelView(points,label, corners,  view, editOrFix=editOrFix, save_path= save_path )#点群を画像に変換

            center_crop_size = (400, 400)
            centerCrop_images(img_paths,  center_crop_size)#画像をクロップ
            image2sketch(img_paths)#画像をスケッチに変換

            #全体画像
            if not os.path.exists(os.path.join(save_path,f"{view}" , "shape_all.png")):
                allPC_path = point2imageWithView(points, corners,view, save_path= save_path, name="shape_all")
                centerCrop_images(allPC_path,  center_crop_size)
                image2sketch(allPC_path)

            tqdmbar.set_postfix(name=name)
        
            # break   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create PartSeg Sketch')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
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
                        default='/home/tomohiro/VAL/Dataset/PointsPartSegWithSketch',
                        help='file format of visualization')
    parser.add_argument('--split', type=str, default='val', metavar='N',
                        choices=['train', 'val'])
    parser.add_argument('--editOrFix', type=str, default='fix', metavar='N',
                        choices=['fix', 'edit'])
    parser.add_argument('--save_path', type=str, default='./output', metavar='N',
                        help='file format of visualization')
    parser.add_argument('--view', type=str, default='X-1Y1Z-1', metavar='N',
                        choices=['X-1Y1Z-1', 'X-0.5Y1Z-1', 'X0Y1Z-1', 'X0.5Y1Z-1', 'X1Y1Z-1', 'X-1Y1Z-0.5', 'X1Y1Z-0.5'])
    parser.add_argument('--save_TF', action='store_true', default=False,)  
    args = parser.parse_args()

    CreatePoint2Sketch(args)