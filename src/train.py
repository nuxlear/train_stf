#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %reload_ext autoreload
# %autoreload 2

import os
import argparse
import ast

import autoencoder as ae
import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
#from torchinfo import summary
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from PIL import Image

from datagen_aug import LipGanDS

from glob import glob
from torch.utils.data import DataLoader, DistributedSampler

from pytorch_warmup import LinearWarmup

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from pathlib import Path

import cv2
import traceback
import sys
from addict import Dict
import pdb


torch.backends.cudnn.benchmark = True
USE_TANH  = True
global_idx = 0


def ddp_setup(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{args.port}'
    print(f'rank:{rank}/{args.n_gpu}', ' port:', os.environ["MASTER_PORT"])
    init_process_group(backend="nccl", init_method="env://", world_size=args.n_gpu, rank=rank)
    print('ddp_setuped')

    
def ddp_clean():
    destroy_process_group()


def save_checkpoint(model, optimizer, log_name, epoch, n_gpu):
    path = f'weights/{log_name[:-3]}/{epoch:03d}.pth'
    print('save checkpoint : ', path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if 1 < n_gpu:
        t = model.module
    else:
        t = model
    checkpoint = {'state_dict':t.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, path)

def invers_transfer(np_image):
    print('1 np_image.shape:', np_image.shape)
    np_image = (np_image + 1.0) *255.0 /2.0
    np_image = np.clip(0, 255, np_image)
    np_image = np_image.astype(np.uint8)
    print('2 np_image.shape:', np_image.shape)
    return np_image


def train_epoch(epoch, model, criterion, optimizer, writer, lr_scheduler, warmup_scheduler, dataloader, device, rank):
    global global_idx
    model.train()
    loss_epoch = 0 
    loss_count = 0
    #for idx, (img_gt, mel, img_gt_masked, _, ips) in enumerate(tqdm(dataloader)):
    for idx, (img_gt, mel, ips) in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        if idx == 0 and rank == 0:
            masked_img = ips[0][:,:,:3].cpu().numpy()
            masked_img = masked_img[:,:,::-1]
            masked_img = invers_transfer(masked_img)
            writer.add_image('train/masked_image', masked_img, epoch, dataformats='HWC')
            
        audio = mel.unsqueeze(1).to(device)
        ips = ips.to(device).permute(0,3,1,2)
        img_gt = img_gt.to(device).permute(0,3,1,2)

        optimizer.zero_grad()

        img_pred = model(ips, audio)

        loss  = criterion(img_pred, img_gt)
        loss_epoch += loss.item()*len(img_gt)
        loss_count += len(img_gt)
        
        loss.backward()

        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
            
    loss_epoch_mean = loss_epoch / loss_count
    if rank == 0:  
        writer.add_scalar('train/loss', loss_epoch_mean, epoch) 
        if lr_scheduler is not None: 
            writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_idx)
        writer.flush()


def now_str():
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def train(rank, args):
    device = torch.device(f'cuda:{rank}')

    def load_model(weight_path=None):
        model = ae.Speech2Face(3, (3, args.img_size, args.img_size), (1, 96, 108)).to(device)
        checkpoint = None
        if weight_path:
            checkpoint = torch.load(weight_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                print('!!!### load model checkpoint["state_dict"]: ', weight_path)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print('!!!### load model checkpoint : ', weight_path)
                model.load_state_dict(checkpoint)
        if 1 < args.n_gpu:
            model = DistributedDataParallel(model, device_ids=[rank])
        return model, checkpoint
    
    model, checkpoint = load_model(args.load_from)

    args.batch_size = args.batch_size_per_gpu * args.n_gpu
    args.num_workers = args.num_workers_per_gpu * args.n_gpu
    
    args.train_images = []
    args.val_images = []
    if not isinstance(args.data_root, (dict)):
        args.data_root = {args.data_root:1}
    for data_root, stride in args.data_root.items():
        args.train_images += sorted(glob(f'{data_root}/train/*/*_yes.jpg'))[::stride]
        args.val_images += sorted(glob(f'{data_root}/val/*/*_yes.jpg'))[::stride]
    # if len(args.train_images) < 10000:
    #     print('invalid len(train_images):', len(args.train_images))
    #     print(args.data_root)
    #     return

    log_name = f'{args.optimizer}_{args.mask_ver}_bs-{args.batch_size}_lr-{args.lr}_mel_ps_{args.mel_ps}_{now_str()}'
    print('log_name:', log_name)
    writer = None
    if rank == 0:
        writer = SummaryWriter(f'runs/{log_name}')
        print(log_name)

    if args.optimizer == 'Adam_Grad':
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=args.lr, eps=1e-7, betas=(0.5, 0.999))

    elif args.optimizer == 'Adam_Base':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)

    elif args.optimizer == 'Adam_Default':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.995, nesterov=True)

    if checkpoint is not None and 'optimizer' in checkpoint:
        print('!!!### load optimizer checkpoint["optimizer"]: ', args.load_from)
        optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint

    criterion = nn.L1Loss()
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
    #                                                steps_per_epoch=len(dl), epochs=args.epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(x) for x in args.milestone.split(',')], last_epoch=-1)
    warmup_scheduler = None
    if isinstance(args.warmup_step, int) and args.warmup_step > 0:
        warmup_scheduler = LinearWarmup(optimizer, warmup_period=args.warmup_step)
    
    ds = LipGanDS(args, phase='train')
    if 1 < args.n_gpu: 
        train_sampler = DistributedSampler(ds, drop_last=True, rank=rank) if args.n_gpu > 1 else None
    else:
        train_sampler = None
    dl = DataLoader(dataset=ds, batch_size=args.batch_size_per_gpu,
                    sampler=train_sampler, shuffle=False if train_sampler else True,
                    pin_memory=True,
                    num_workers=args.num_workers_per_gpu,
                    drop_last=True)
    for epoch in tqdm(range(args.total_epochs)): 
        
        #if 0 == rank:
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        train_epoch(epoch, model, criterion, optimizer, writer, scheduler, warmup_scheduler, dl, device, rank)

        if train_sampler:
            dist.barrier()
        
        if 0 == rank:
            save_checkpoint(model, optimizer, log_name=log_name, epoch=epoch, n_gpu=args.n_gpu)
            
        if train_sampler: 
            dist.barrier()


def train_(rank, args):
    if 1 < args.n_gpu:
        ddp_setup(rank, args)

    try:
        train(rank, args)
    except:
        traceback.print_exception(*sys.exc_info())
    finally:
        if 1 < args.n_gpu:
            ddp_clean()

def main(args):
    print('n_gpu:', args.n_gpu)

    if args.n_gpu > 1:
        mp.spawn(train_, nprocs=args.n_gpu, args=(args,), join=True)
    else:
        train_(0, args)



def arg_parse():
    parser = argparse.ArgumentParser(description='torch implementation of voice_to_lip')
    parser.add_argument('--mel_ps', type=int, default=80) 
    parser.add_argument('--mel_step_size', type=int, default=108)
    parser.add_argument('--mask_img_trsf_ver', type=int, default=0)
    parser.add_argument('--mask_shape_rand_add_scale', type=float, default=0.3)
    parser.add_argument('--num_ips', type=int, default=2)
    parser.add_argument('--total_epochs', type=int, default=300)
    parser.add_argument('--mel_trsf_ver', type=int, default=-1)
    parser.add_argument('--mel_norm_ver', type=int, default=-1)
    
    
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--batch_size_per_gpu', type=int, default=16)
    parser.add_argument('--mask_ver', type=str, default='(9)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_root', type=str, default='{"data_root":1}')
    parser.add_argument('--num_workers_per_gpu', type=int, default=9)
    
    parser.add_argument('--optimizer', type=str, default='Adam_Default')
    parser.add_argument('--warmup_step', type=int, default=5)
    parser.add_argument('--milestone', type=str, default='100,150')
    parser.add_argument('--load_from', type=str, default=None)
    
    args = parser.parse_args()
    
    args.data_root = ast.literal_eval(args.data_root)
    args.mask_ver = ast.literal_eval(args.mask_ver)
    
    print(args)
    
    return args
        
if __name__ == "__main__":

    master_port = 70000 + np.random.randint(0, 10000)
    args = arg_parse()
    
    args.n_gpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    print('!!!### n_gpu : ', args.n_gpu)
    args.port = master_port

    print('master port:', args.port)

    main(args)
