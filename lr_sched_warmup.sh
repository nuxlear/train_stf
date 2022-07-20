PYTHON_='/root/miniconda3/envs/stf/bin/python'

$PYTHON_ -m pip install tensorboardX
$PYTHON_ -m pip install pytorch_warmup --no-deps

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON_ src/train.py \
	--data_root "{'./nuxlear_lr_scheduler':1}" \
	--img_size 352 \
	--mask_ver "('pwb_front_v39_1')" \
	--total_epochs 300 \
	--optimizer SGD \
	--lr 0.001 \
	--warmup_step 5 \
	--milestone 100,150 \
	--batch_size_per_gpu 24 \
	--num_workers_per_gpu 8 






