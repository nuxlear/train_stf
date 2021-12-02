CUDA_VISIBLE_DEVICES=1,2,3 python src/train.py \
	--data_root "{'./data_root_front':1, './data_root_front_mon_wed/':1}" \
	--img_size 352 \
	--mask_ver  "('pwb_front_v39_1')" \
	--total_epochs 300 \
	--optimizer Adam_Default \
	--lr 0.0001 \
	--batch_size_per_gpu 24 \
	--num_workers_per_gpu 8 \
    --load_from "/home/kts123/aia/pwb/s2f_torch/weights/Adam_Default_pwb_front_v39_0_bs-72_lr-0.0001_mel_ps_80_2021-09-27 22:51/042.pth"
























