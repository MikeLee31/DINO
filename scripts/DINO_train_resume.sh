coco_path=$1
ck_path=$2
start_epoch=$3
python main.py \
	--output_dir logs/DINO/R50-MS4 -c config/DINO/DINO_4scale.py --coco_path $coco_path \
	--resume $ck_path \
	--start_epoch $start_epoch \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
