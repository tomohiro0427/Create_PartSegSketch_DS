v2_path="../Dataset/ShapeNetCore.v2.PC15k"
output_path="../Dataset/5_views_partSeg_pyrender_chair_airplane_table/"
echo "${v2_path}"
echo "${output_path}"

python npy_label_save.py --split val --output_path ${output_path} --dataset_path ${v2_path} --categories chair
python npy_label_save.py --split train --output_path ${output_path} --dataset_path ${v2_path} --categories chair

python npy_label_save.py --split val --output_path ${output_path} --dataset_path ${v2_path} --categories airplane
python npy_label_save.py --split train --output_path ${output_path} --dataset_path ${v2_path} --categories airplane

python npy_label_save.py --split val --output_path ${output_path} --dataset_path ${v2_path} --categories table
python npy_label_save.py --split train --output_path ${output_path} --dataset_path ${v2_path} --categories table

# python npy_label_save.py --split val --output_path ${output_path} --dataset_path ${v2_path} --categories car
# python npy_label_save.py --split train --output_path ${output_path} --dataset_path ${v2_path} --categories car


python pyrender_create_dataset.py --dataset_path ${output_path} --split val --num_view 5 --class_choice chair
python pyrender_create_dataset.py --dataset_path ${output_path} --split val --num_view 5 --class_choice airplane
python pyrender_create_dataset.py --dataset_path ${output_path} --split val --num_view 5 --class_choice table
# python pyrender_create_dataset.py --dataset_path ${output_path} --split val --num_view 5 --class_choice car

python pyrender_create_dataset.py --dataset_path ${output_path} --split train --num_view 5 --class_choice airplane
python pyrender_create_dataset.py --dataset_path ${output_path} --split train --num_view 5 --class_choice chair
python pyrender_create_dataset.py --dataset_path ${output_path} --split train --num_view 5 --class_choice table
# python pyrender_create_dataset.py --dataset_path ${output_path} --split train --num_view 5 --class_choice car