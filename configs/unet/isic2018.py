_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/isic2018.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=4000, val_interval=400)