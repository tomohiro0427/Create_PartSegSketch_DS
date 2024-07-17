# #FULLMOON
# v2_path="/home/tomohiro/VAL/Train_CanonicalVAE/mini_dataset_3"
# output_path="/home/tomohiro/VAL/Dataset/PointsPartSegWithSketch"
# echo "${v2_path}"
# echo "${output_path}"

# python npy_label_save.py --split val --output_path ${output_path} --dataset_path ${v2_path} --categories chair

# python create_PartSeg_Sketch.py  --split val  --editOrFix fix --view X-1Y1Z-1 --dataset_path ${output_path}  --class_choice chair
# python create_PartSeg_Sketch.py  --split val  --editOrFix edit --view X-1Y1Z-1 --dataset_path ${output_path} --class_choice chair

#Europe
# v2_path="/home/tomohiro/workspace/Dataset/ShapeNetCore.v2.PC15k"
# output_path="/home/tomohiro/workspace/Dataset/PointsPartSegWithSketch"
v2_path="/home/tomohiro/workspace/Dataset/v2.mini"
output_path="/home/tomohiro/workspace/Dataset/PartSegSketch_mini"
echo "${v2_path}"
echo "${output_path}"

python npy_label_save.py --split val --output_path ${output_path} --dataset_path ${v2_path} --categories chair
# python npy_label_save.py --split train --output_path ${output_path} --dataset_path ${v2_path} --categories chair

# python create_PartSeg_Sketch.py  --split train  --editOrFix fix --view X-1Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split train  --editOrFix fix --view X-0.5Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split train  --editOrFix fix --view X0Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split train  --editOrFix fix --view X0.5Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split train  --editOrFix fix --view X1Y1Z-1 --dataset_path ${output_path} --class_choice chair

# python create_PartSeg_Sketch.py  --split train  --editOrFix edit --view X-1Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split train  --editOrFix edit --view X-0.5Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split train  --editOrFix edit --view X0Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split train  --editOrFix edit --view X0.5Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split train  --editOrFix edit --view X1Y1Z-1 --dataset_path ${output_path} --class_choice chair



python create_PartSeg_Sketch.py  --split val  --editOrFix fix --view X-1Y1Z-1 --dataset_path ${output_path} --class_choice chair
python create_PartSeg_Sketch.py  --split val  --editOrFix fix --view X-0.5Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split val  --editOrFix fix --view X0Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split val  --editOrFix fix --view X0.5Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split val  --editOrFix fix --view X1Y1Z-1 --dataset_path ${output_path} --class_choice chair

python create_PartSeg_Sketch.py  --split val  --editOrFix edit --view X-1Y1Z-1 --dataset_path ${output_path} --class_choice chair
python create_PartSeg_Sketch.py  --split val  --editOrFix edit --view X-0.5Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split val  --editOrFix edit --view X0Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split val  --editOrFix edit --view X0.5Y1Z-1 --dataset_path ${output_path} --class_choice chair
# python create_PartSeg_Sketch.py  --split val  --editOrFix edit --view X1Y1Z-1 --dataset_path ${output_path} --class_choice chair
