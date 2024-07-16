
# Pairs of Point Cloud PartSeg Labels and Sketch Images
A dataset that creates pairs of point clouds and sketches for each part based on PartSegmentation results by [DGCNN](https://github.com/antao97/dgcnn.pytorch/tree/master)

## Requirements
python ==3 .10<br>
pytorch == 2.3.1　<br>
cuda == 11.8　<br>

plyfile,
pandas,
open3d,
opencv-python,
pyarrow,
fastparquet,
ninja,

---

Here are the steps to create the requirements environment:

```
conda create -n env_name python=3.10
```
Install [pytorch](https://pytorch.org/)
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

requirements.txt
```
pip install -r requirements.txt
```

Install PointNet++ for FPS
```
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```
Install CUDA KNN
```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Download ShapeNetV2 (Point Cloud Dataset)
We use the processed version of ShapeNet provided [PointFlow](https://github.com/stevenygd/PointFlow).

## Create Dataset
Here are the steps to run the shell script:

1\.Open the Make_dataset_partseg.bash file in a text editor.

2\.Locate the v2_path and output_path variables in the script.

3\.Replace the placeholder paths with the actual paths on your system where you have the ShapeNetV2 data and where you want to save the output.
```Shell:Make_dataset_partseg.bash
v2_path="your own path of dataset(ShapeNetCore.v2.PC15k)"
output_path="your own path of output"
```
Run the Shell script(Make_dataset_partseg.bash)
```
bash Make_dataset_partseg.bash
```

## Created Dataset Structure
```
PcWithSketchDataset/
├── train/
│   ├── 0300167/ [Class]
│   │   ├── 1ace72an/ [One Pc data]
│   │   │   ├── 1ace72an.npy
│   │   │   ├── 1ace72an.parquest
│   │   │   │
│   │   │   ├── X0.5Y1Z-1/ [view]
│   │   │   │   ├── shape_all.png
│   │   │   │   ├── shape_all_sketch.png
│   │   │   │   │
│   │   │   │   ├── 12/ [Part label]
│   │   │   │   │   ├── edit.png
│   │   │   │   │   ├── edit_sketch.png
│   │   │   │   │   ├── fix.png
│   │   │   │   │   └── fix_sketch.png
│   │   │   │   │
│   │   │   │   └── 13/ [Part label]
│   │   │   │        ├── ...
│   │   │   │        ├── ...
│   │   │   │    
│   │   │   └── X0Y1Z-1/ [view]
│   │   │        ├── ...
│   │   │        ├── ...
│   │   │  
│   │   └── 2afa06.../ [One Pc data]
│   │        ├── ....
│   │        ├── ....
│   │   
│   └── 123455/ [Class]
│        ├── ....
│        ├── ....
│    
└── val/
    ├── ....

```