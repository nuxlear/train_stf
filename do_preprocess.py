import preprocess


seed = 42

clips = '/data/hantu_data/preprocessed/*/*.pickle'
data_root = 'nuxlear_lr_scheduler'
val_count = 10

r = preprocess.get_clip_count(clips, val_count, seed=seed)
preprocess.make(data_root=data_root, **r)
