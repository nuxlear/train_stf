CUDA_VISIBLE_DEVICES=4,5,6,7 python src/train.py \
	--data_root "{'../stf_data/data_root_front':1}" \
	--img_size 352 \
	--mask_ver  "('pwb_front_v39_1')" \
	--total_epochs 300 \
	--optimizer Adam_Default \
	--lr 0.0001 \
	--batch_size_per_gpu 24 \
	--num_workers_per_gpu 8 
