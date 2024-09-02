v2_path="../Dataset/ShapeNetCore.v2.PC15k"
output_path="../Dataset/PartSeg_pyrender_unet/"
echo "${v2_path}"
echo "${output_path}"

python npy_label_save.py --split val --output_path ${output_path} --dataset_path ${v2_path} --categories chair
python npy_label_save.py --split train --output_path ${output_path} --dataset_path ${v2_path} --categories chair

python pyrender_create_dataset.py --dataset_path ${output_path} --split val --num_view 4
python pyrender_create_dataset.py --dataset_path ${output_path} --split train --num_view 14