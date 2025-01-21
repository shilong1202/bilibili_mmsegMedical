_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/segpc.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=3,
        # loss_decode=[
        #     dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        #     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]
    ))

# optimizer=dict(type='AdamW', lr=0.01, betas=(0.9, 0.999), weight_decay=0.05)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
#     dict(
#         type='PolyLR',
#         power=1.0,
#         begin=1000,
#         end=40000,
#         eta_min=0.0,
#         by_epoch=False,
#     )
# ]

