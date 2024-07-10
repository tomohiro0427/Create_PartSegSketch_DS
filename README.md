
# 点群のPartSegmentationラベルとスケッチ画像のペアデータセット
DGCNNによるPartSegmentationをした結果をもとに部位ごとにスケッチ化した画像と点群のペアを作成するデータセット

## 推奨環境
pytorch11.8

```
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## ShapeNetV2のダウンロード
以下のリンクから点群のデータをダウンロードしてください　

## データセット作成
Make_dataset_partseg.bashと’v2_path’と'output_path'を自分用に書き換えてください
```
bash Make_dataset_partseg.bash
```