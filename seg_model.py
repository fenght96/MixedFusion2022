import os

import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pdb

matplotlib.use('AGG')


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imsave(name + '.png', image.astype('uint8'))
    plt.show()

def visualize_mask(name, image):
    """PLot images in one row."""
    plt.figure(figsize=(16, 5))
    
    plt.xticks([])
    plt.yticks([])
    plt.title(' '.join(name.split('_')).title())
    plt.imsave(name + '.png', image * 255)
    plt.show()

class PoseDataset(Dataset):
    CLASSES = ['back1','glue_1', 'square_plate_1', 'suger_2', 'potato_chip_2', 'small_clamp', 'lipton_tea', 'phillips_screwdriver', 'book_1',
            'round_plate_1', 'orion_pie', 'plate_holder', 'round_plate_4', 'book_2', 'plastic_banana', 'power_drill', 'round_plate_2',
            'potato_chip_1', 'potato_chip_3', 'square_plate_3', 'square_plate_4', 'large_clamp', 'glue_2', 'extra_large_clamp', 'suger_1',
            'round_plate_3', 'large_marker', 'medium_clamp', 'flat_screwdriver', 'scissors', 'book_3', 'correction_fuid', 'square_plate_2',
            'mini_claw_hammer_1']
    def __init__(self, root, mode, augmentation, preprocessing):
        if mode == 'train':
            self.path = 'datasets/ycb/dataset_config/train.txt'
        elif mode == 'test':
            self.path = 'datasets/ycb/dataset_config/test.txt'
        self.root = root
        self.list = []
        with open(self.path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            self.list.append(line.strip().replace('/home/casia/code/icra/','/home/fht/data/'))

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.class_values = [i for i in range(1, len(self.CLASSES))]

    def __getitem__(self, index):

        # read data
        img = cv2.imread('{0}.png'.format(self.list[index]))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.load('{0}.npy'.format(self.list[index].replace('rgb_undistort', 'seg_masks')))
        mask = mask[:,:,np.newaxis]
        
        masks = [(mask == v) for v in self.class_values]

        mask = np.stack(masks, axis=-1).astype('float')
        print(mask.shape)
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        print(mask.shape)


        return image, mask

    def __len__(self):
        return len(self.list)

def get_training_augmentation():
    train_transform = [

        # albu.HorizontalFlip(p=0.5),


        albu.PadIfNeeded(min_height=736, min_width=1280, always_apply=True, border_mode=0),
        albu.RandomCrop(height=352, width=640, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        albu.PadIfNeeded(736, 1280)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


if __name__ == '__main__':
	
    root = '/home/fht/data/ocrtoc/'
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'


    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS, 
        classes=33,
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
	
    train_dataset = PoseDataset(
        root=root,
        mode='train',
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    # for j in range(len(train_dataset)):
    #     img, mask, ori_img, ori_mask = train_dataset[j]
    #     visualize(img=ori_img, mask=ori_mask)
    #     for i in range(mask.shape[0]):
    #         visualize_mask('mask_'+ str(i) ,mask[i])
    #     pdb.set_trace()

    valid_dataset = PoseDataset(
        root=root,
        mode='test',
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    loss = smp.losses.FocalLoss(mode='multiclass')
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])
    # optimizer = torch.optim.SGD([
    #     dict(params=model.parameters(), lr=0.01),
    # ])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_score = 0

    for i in range(0, 40):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        val_logs = valid_epoch.run(valid_loader)
		
		# 每次迭代保存下训练最好的模型
        if max_score < val_logs['iou_score']:
            max_score = val_logs['iou_score']
            torch.save(model, './with_back_seg_best_model.pth')
            print('Model saved!')
            print(f'max score:{max_score:.5f}')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

