dataest_type = 'Animal5'
# below are not acctually mean/std of flowers102
# https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
dataset_mean = (0.485, 0.456, 0.406)
dataset_std = (0.229, 0.224, 0.225)
max_epochs = 20
batch_size = 32
lr = 0.1 / 256 * batch_size
num_workers = 0
pin_memory = False

trainer = dict(
    type='EpochNormalTrainer',
    nos=dict(type='SimpleFreezeNOS',
             network=dict(arch='torchvision',
                          type='mobilenet_v2',
                          weights='DEFAULT',
                          progress=True),
             optimizer=dict(type='SGD',
                            lr=0.01,
                            momentum=0.9,
                            weight_decay=1e-5),
             scheduler=dict(type='CosineAnnealingLR', T_max=20),
             training=dict(mode=True, trainable_modules=['classifier.1']),
             mapping={
                 'classifier.1':
                 dict(type='Linear', in_features=1280, out_features=5)
             }),
    train_dataloader=dict(dataset=dict(type=dataest_type,
                                       root='data/Animal Classification',
                                       split='train',
                                       transform=[
                                           dict(type='RandomRotation',
                                                degrees=30),
                                           dict(type='RandomResizedCrop',
                                                size=224),
                                           dict(type='RandomHorizontalFlip'),
                                           dict(type='ToTensor'),
                                           dict(type='Normalize',
                                                mean=dataset_mean,
                                                std=dataset_std)
                                       ]),
                          batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          shuffle=True),
    val_dataloader=dict(dataset=dict(type=dataest_type,
                                     root='data/Animal Classification',
                                     split='val',
                                     transform=[
                                         dict(type='Resize', size=(256, 256)),
                                         dict(type='CenterCrop', size=224),
                                         dict(type='ToTensor'),
                                         dict(type='Normalize',
                                              mean=dataset_mean,
                                              std=dataset_std)
                                     ]),
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory),
    test_dataloader=dict(dataset=dict(type=dataest_type,
                                      root='data/Animal Classification',
                                      split='test',
                                      transform=[
                                          dict(type='Resize', size=(256, 256)),
                                          dict(type='CenterCrop', size=224),
                                          dict(type='ToTensor'),
                                          dict(type='Normalize',
                                               mean=dataset_mean,
                                               std=dataset_std)
                                      ]),
                         batch_size=batch_size,
                         num_workers=num_workers,
                         pin_memory=pin_memory),
    epochs=max_epochs,
    save_interval=1,
    device='cpu')
