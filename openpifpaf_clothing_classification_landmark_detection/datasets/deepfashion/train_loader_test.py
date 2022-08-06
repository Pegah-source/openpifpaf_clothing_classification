# this file is only used to test the preprocessing, without many elements added in the datamodule
import argparse

import torch
import openpifpaf

from .dataset import DeepfashionDataset
from . import transforms



from .constants import (
    DEEPFASHION_SKELETON,
    DEEPFASHION_KEYPOINTS,
    DEEPFASHION_POSE,
    HFLIP,
    DEEPFASHION_SIGMAS,
    DEEPFASHION_SCORE_WEIGHTS,
    DEEPFASHION_CATEGORIES
)

'''
    this is the part of the code where we gather all the elements that help openpifpaf run train with the specific setup. 
    Preprocessing and dataloader are defined here.
'''
class Deepfashion_module(openpifpaf.datasets.DataModule):
    """DataModule for Deepfashion dataset."""

    debug = False
    pin_memory = False

    # General
    train_annotations = '../../clean_trial/json_annos/train_annotations_MSCOCO_style.json'
    val_annotations = '../../clean_trial/json_annos/valid_annotations_MSCOCO_style.json'
    eval_annotations = '../../clean_trial/json_annos/test_annotations_MSCOCO_style.json'
    train_image_dir = '../../clean_trial/json_annos/img_train'
    val_image_dir = '../../clean_trial/json_annos/img_valid'
    eval_image_dir = '../../clean_trial/json_annos/img_test'

    # Tasks
    taks_group = ['classification']
    upsample_stride = 1

    # Train Pre-processing
    square_edge = 350
    with_dense = False
    augmentation = True

    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.4
    augmentation_level = 1
    rescale_images = 1.0
    min_kp_anns = 1
    bmin = 0.1

    skeleton = DEEPFASHION_SKELETON


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module deepfashionkp')

        group.add_argument('--deepfashion-train-annotations', default=cls.train_annotations,
                           help='train annotations')
        group.add_argument('--deepfashion-val-annotations', default=cls.val_annotations,
                           help='val annotations')
        group.add_argument('--deepfashion-train-image-dir', default=cls.train_image_dir,
                           help='train image dir')
        group.add_argument('--deepfashion-val-image-dir', default=cls.val_image_dir,
                           help='val image dir')

        group.add_argument('--deepfashion-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        group.add_argument('--deepfashion-attributes',
                           default=cls.taks_group, nargs='+',
                           help='list of attributes to consider for clothings')
        assert not cls.with_dense
        group.add_argument('--deepfashion-with-dense',
                           default=False, action='store_true',
                           help='train with dense connections')
        assert not cls.extended_scale
        group.add_argument('--deepfashion-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--deepfashion-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--deepfashion-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--deepfashion-augmentation-level',
                           default=1, type = int,
                           help='level of augmentation')
        group.add_argument('--deepfashion-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--deepfashion-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--deepfashion-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--deepfashion-bmin',
                           default=cls.bmin, type=float,
                           help='bmin')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # Extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # General
        cls.train_annotations = args.deepfashion_train_annotations
        cls.val_annotations = args.deepfashion_val_annotations
        cls.train_image_dir = args.deepfashion_train_image_dir
        cls.val_image_dir = args.deepfashion_val_image_dir
        cls.batch_size = args.batch_size


        # Pre-processing
        cls.square_edge = args.deepfashion_square_edge
        cls.with_dense = args.deepfashion_with_dense
        cls.extended_scale = args.deepfashion_extended_scale
        cls.orientation_invariant = args.deepfashion_orientation_invariant
        cls.blur = args.deepfashion_blur
        cls.augmentation_level = args.deepfashion_augmentation_level
        cls.rescale_images = args.deepfashion_rescale_images
        cls.upsample_stride = args.deepfashion_upsample
        cls.min_kp_anns = args.deepfashion_min_kp_anns
        cls.bmin = args.deepfashion_bmin


    def _common_preprocess_op(self):
        # a bbox cropping should be added here!!!!
        return [
            transforms.BBoxCrop(),
            transforms.CheckLandmarks(),
            transforms.NormalizeAnnotations(),
            openpifpaf.transforms.RescaleAbsolute(350),
            openpifpaf.transforms.CenterPad(350)
        ]


    def _train_preprocess(self):
        if self.augmentation_level == 0:

            data_augmentation_op = [transforms.EVAL_TRANSFORM]
            
        elif self.augmentation_level == 1:

            data_augmentation_op = [
                openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(DEEPFASHION_KEYPOINTS, HFLIP), 0.5),
                openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
                # openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.TRAIN_TRANSFORM
            ]

        elif self.augmentation_level == 2:

            data_augmentation_op = [
                openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(DEEPFASHION_KEYPOINTS, HFLIP), 0.5),
                openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
                openpifpaf.transforms.RandomChoice(
                [openpifpaf.transforms.RotateUniform(20.0)],
                [0.4],
            ),
                # openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.TRAIN_TRANSFORM
            ]            


        return openpifpaf.transforms.Compose([
            *self._common_preprocess_op(),
            *data_augmentation_op,
        ])
    
    def train_loader(self):
        train_data = DeepfashionDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._train_preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns
        )
        
        return torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=not self.debug and self.augmentation,
            num_workers=0,
            drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta,
        )