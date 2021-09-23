CUDA_VISIBLE_DEVICES=1,2,3 python src/train.py \
	--data_root "{'./data_root_front':1}" \
	--img_size 352 \
	--mask_ver  "(9)" \
	--total_epochs 300 \
	--optimizer Adam_Default \
	--lr 0.0001 \
	--batch_size_per_gpu 24 \
	--num_workers_per_gpu 8 \
    --load_from "./weights/Adam_Default_9_bs-72_lr-0.0001_mel_ps_80_2021-09-14 01:31/155.pth"
























