from src.dataset_module import module_dataset
from torch.utils.data import DataLoader
import open3d as o3d
import matplotlib.pyplot as plt


dataset = module_dataset.PointCloudWithPartSegSketch(
                                                        root='/home/tomohiro/VAL/Dataset/PointsPartSegWithSketch',
                                                        split='val',
                                                        categories=['chair'],
                                                        get_images = ['edit_sketch','fix_image'],
                                                    )

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, batch in enumerate(dataloader):
    label = batch.get('label')# [B, N]
    points = batch.get('point_cloud')# [B, N, 3]
    edit_sketchs = batch.get('edit_sketch')# [B, N, 3]
    fix_sketchs = batch.get('fix_image')# [B, N, 3]
    name = batch.get('name')
    print(edit_sketchs.shape)
    print(name)

    save_path = './output'
    print(label.shape)
    print(points.shape)
    

    # Assuming edit_sketchs is a tensor of shape [B, N, 3]
    for sketch in edit_sketchs:
        plt.imshow(sketch.permute(1, 2, 0))
        plt.show()
    for sketch in fix_sketchs:
        plt.imshow(sketch.permute(1, 2, 0))
        plt.show()
    # # Assuming points is a tensor of shape [B, N, 3]
    # for point_cloud in points:
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(point_cloud)
    #     o3d.visualization.draw_geometries([pcd])


    break