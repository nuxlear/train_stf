# 훈련 데이터 준비
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm.auto import tqdm
import random
import shutil
import os


def get_clip_count(clip_name, val_cnt, seed=1234):
    clips = glob(clip_name)
    print('total clip count :', len(clips))
    clips = [Path(e).parent for e in clips]
    random.seed(seed)
    random.shuffle(clips)
    if len(clips) <= val_cnt:
        print('val_cnt 가 너무 큽니다., val_cnt:', val_cnt, ', total clip:', len(clips))
        return
    clip_val = clips[:val_cnt]
    clip_train = list(set(clips) - set(clip_val))
    print('파일을 하나씩 보여줍니다.')
    print('train:', clip_train[0])
    print('val:', clip_val[0])
    print('train count :', len(clip_train), ', valcount:', len(clip_val))
    return {'clip_train':clip_train, 'clip_val':clip_val}


def make(data_root, clip_train, clip_val):
    # data_root 폴더를 삭제하고 다시 만든다
    shutil.rmtree(data_root, ignore_errors=True)
    Path(f'{data_root}/train').mkdir(parents=True)
    Path(f'{data_root}/val').mkdir(parents=True)

    # link 걸기
    def make_link(base, targets):
        for target in tqdm(targets):
            link = f'{base}/{Path(target).name}'
            #cmd = f'ln -s "{target}" "{link}"'
            #!$cmd
            os.symlink(target, link)

    make_link(f'{data_root}/train',  clip_train)
    make_link(f'{data_root}/val',    clip_val)