#CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py \
#	--data_root "{'./pwb_mon_front':1}" \
#	--img_size 352 \
#	--mask_ver  "(9)" \
#	--total_epochs 300 \
#	--optimizer Adam_Default \
#	--lr 0.0001 \
#	--batch_size_per_gpu 24 \
#	--num_workers_per_gpu 8 


CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py \
	--data_root "{'./pwb_mon_front':1}" \
	--img_size 352 \
	--mask_ver  "('pwb_front_v39_1')" \
	--total_epochs 300 \
	--optimizer Adam_Default \
	--lr 0.0001 \
	--batch_size_per_gpu 24 \
	--num_workers_per_gpu 8 



















