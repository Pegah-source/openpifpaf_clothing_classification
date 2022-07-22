import argparse

import torch
import openpifpaf

from .attribute import DeepfashionType
from .dataset import DeepfashionDataset
from . import transforms
from .. import annotation
from .. import attribute
from .. import encoder
from .. import headmeta
from .. import metrics as eval_metrics
from .. import sampler


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
    train_annotations = '../../../../scratch/izar/khayatan/deepfashion/train_annotations_MSCOCO_style.json'
    val_annotations = '../../../../scratch/izar/khayatan/deepfashion/valid_annotations_MSCOCO_style.json'
    eval_annotations = '../../../../scratch/izar/khayatan/deepfashion/test_annotations_MSCOCO_style.json'
    train_image_dir = '../../../../scratch/izar/khayatan/deepfashion/img_train'
    val_image_dir = '../../../../scratch/izar/khayatan/deepfashion/img_valid'
    eval_image_dir = '../../../../scratch/izar/khayatan/deepfashion/img_test'
    # /pegah/vita/work/../scratch

    # Tasks
    taks_group = ['classification']
    upsample_stride = 1

    # Train Pre-processing
    square_edge = 300
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



    def __init__(self):
        super().__init__()
        self.compute_attributes()
        self.compute_head_metas()


    @classmethod
    def compute_attributes(cls):
        cls.attributes = {
            DeepfashionType.Clothing: cls.taks_group,
        }


    @classmethod
    def compute_head_metas(cls):
        att_metas = attribute.get_attribute_metas(dataset='deepfashion',
                                                  attributes=cls.attributes)
        cif = openpifpaf.headmeta.Cif('cif', 'deepfashion',
                                      keypoints=DEEPFASHION_KEYPOINTS,
                                      sigmas=DEEPFASHION_SIGMAS,
                                      pose=DEEPFASHION_POSE,
                                      draw_skeleton=cls.skeleton,
                                      score_weights=DEEPFASHION_SCORE_WEIGHTS)
        caf = openpifpaf.headmeta.Caf('caf', 'deepfashion',
                                      keypoints=DEEPFASHION_KEYPOINTS,
                                      sigmas=DEEPFASHION_SIGMAS,
                                      pose=DEEPFASHION_POSE,
                                      skeleton=cls.skeleton)

        cif.upsample_stride = cls.upsample_stride
        caf.upsample_stride = cls.upsample_stride
        # cif and caf should also be added here
        # the object **am does also contain the object_type.
        cls.head_metas = [headmeta.ClassMeta('attribute-'+am['attribute'],
                                                 'deepfashion', **am)
                          for am in att_metas]
        cls.head_metas.append(cif)
        cls.head_metas.append(caf)
        for hm in cls.head_metas:
            hm.upsample_stride = cls.upsample_stride


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module deepfashionkp')

        group.add_argument('--deepfashion-train-annotations', default=cls.train_annotations,
                           help='train annotations')
        group.add_argument('--deepfashion-val-annotations', default=cls.val_annotations,
                           help='val annotations')
        group.add_argument('--deepfashion-test-annotations', default=cls.eval_annotations,
                           help='test annotations')
        group.add_argument('--deepfashion-train-image-dir', default=cls.train_image_dir,
                           help='train image dir')
        group.add_argument('--deepfashion-val-image-dir', default=cls.val_image_dir,
                           help='val image dir')
        group.add_argument('--deepfashion-test-image-dir', default=cls.eval_image_dir,
                           help='test image dir')

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
        cls.test_annotations = args.deepfashion_test_annotations
        cls.train_image_dir = args.deepfashion_train_image_dir
        cls.val_image_dir = args.deepfashion_val_image_dir
        cls.test_image_dir = args.deepfashion_test_image_dir

        # Tasks
        cls.taks_group = args.deepfashion_attributes
        cls.compute_attributes()
        cls.upsample_stride = args.deepfashion_upsample
        cls.compute_head_metas()

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
                transforms.TRAIN_TRANSFORM
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
                transforms.TRAIN_TRANSFORM
            ]
        
        encoders = [encoder.AttributeEncoder(self.head_metas[0])]
        encoders.append(openpifpaf.encoder.Cif(self.head_metas[1], bmin=self.bmin))
        encoders.append(openpifpaf.encoder.Caf(self.head_metas[2], bmin=self.bmin))

        return openpifpaf.transforms.Compose([
            *self._common_preprocess_op(),
            *data_augmentation_op,
            openpifpaf.transforms.Encoders(encoders)
        ])

    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *self._common_preprocess_op(),
            transforms.ToAnnotations(annotation.OBJECT_ANNOTATIONS['deepfashion']),
            transforms.EVAL_TRANSFORM,
        ]) # RE-check this ToAnnotations thing, considering the fact that we have combined_annotations

    
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

    def val_loader(self):
        val_data = DeepfashionDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._train_preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns
        )
        return torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=(not self.debug) and self.augmentation,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta,
        )


    def eval_loader(self):
        eval_data = DeepfashionDataset(
            image_dir=self.test_image_dir,
            ann_file=self.test_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns
        )
        return torch.utils.data.DataLoader(
            eval_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta,
        )

    def metrics(self):
        return [eval_metrics.Deepfashion_landmark(self.test_annotations),
                eval_metrics.Deepfashion_classification(self.test_annotations)]
