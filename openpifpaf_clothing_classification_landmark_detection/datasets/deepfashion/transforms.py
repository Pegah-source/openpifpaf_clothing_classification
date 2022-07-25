import copy
import logging
from turtle import right
import warnings

import numpy as np
import openpifpaf
import PIL
import torch
import matplotlib.pyplot as plt



from .attribute import DeepfashionType
from .. import annotation


LOG = logging.getLogger(__name__)


class NormalizeAnnotations(openpifpaf.transforms.Preprocess):
    @staticmethod
    def normalize_annotations(anns):
        anns = copy.deepcopy(anns)

        for ann in anns:# in our case there is only one annotation
            if isinstance(ann, openpifpaf.annotation.Base):
                # Already converted to an annotation type
                continue
            # the problem I am facing here is that ann does not seem to have any attr as object_type .... where do we define this attribute for jaad??
            # what are the annotations like?? are they an instance of .... actually it is indicated in the dataset itself
            if ann['object_type'] is DeepfashionType.Clothing:
            
                if 'keypoints' not in ann:
                    ann['keypoints'] = []
                if 'iscrowd' not in ann:
                    ann['iscrowd'] = False

                ann['keypoints'] = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                
                if 'segmentation' in ann:
                    del ann['segmentation']
                ann['object_type'] = -1

        return anns


    def __call__(self, image, anns, meta):
        anns = self.normalize_annotations(anns)

        if meta is None:
            meta = {}

        # fill meta with defaults if not already present
        w, h = image.size
        meta_from_image = {
            'offset': np.array((0.0, 0.0)),
            'scale': np.array((1.0, 1.0)),
            'rotation': {'angle': 0.0, 'width': None, 'height': None},
            'valid_area': np.array((0.0, 0.0, w - 1, h - 1)),
            'original_left' : 0,
            'original_top' : 0,
            'hflip': False,
            'width_height': np.array((w, h)),
        }
        for k, v in meta_from_image.items():
            if k not in meta:
                meta[k] = v

        return image, anns, meta

# added this function to crop out the bbox of the image
# should also check for the keypoints that might be now outside of the image
class BBoxCrop(openpifpaf.transforms.Preprocess):

    def __call__(self, image, anns, meta):
        # croping the bbox, first preprocessing
        # so the image should be like x and y and bbox is like the same
        #exp_image = np.transpose(image, (1, 2, 0))
        '''exp_image = np.array(image)
        exp_image = torch.tensor(exp_image, dtype = float)
        print('this is the size ', exp_image.size())
        # first standardize the data
        mean_exp_image = torch.mean(exp_image)
        std_exp_image = torch.std(exp_image)
        exp_image = (exp_image - mean_exp_image)/std_exp_image
        exp_image = ((exp_image-torch.min(exp_image))*255.0)/(torch.max(exp_image)-torch.min(exp_image))
        
        exp_image = exp_image.int()
        plt.imshow(exp_image)
        plt.show()'''
        print('cropping the bbox, first step of preprocessing')

        landmarks = anns[0]['keypoints']
        x_1 = anns[0]['bbox'][0]
        y_1 = anns[0]['bbox'][1]
        
        top = y_1
        left = x_1
        meta['original_left'] = left
        meta['original_top'] = top
        new_h = anns[0]['bbox'][3]
        new_w = anns[0]['bbox'][2]

        im_np = np.asarray(image)
        im_np = im_np[top: top + new_h, left: left + new_w]
        image = PIL.Image.fromarray(im_np)
        
        
        # some landmarks may even become negative with the following operation
        temp_sides = [left, top]

        def is_coordinate(index):
            if (index+1)%3 > 0:
                return 1
            else:
                return 0
        landmarks = [(landmarks[3*i+j] - temp_sides[j%2]*(is_coordinate(j))) for i in range(8) for j in range(3)]
        anns[0]['keypoints'] = landmarks


        return image, anns, meta

class CheckLandmarks(openpifpaf.transforms.Preprocess):

    def __call__(self, image, anns, meta):
        w, h = image.size
        landmarks = (anns[0]['keypoints']).copy()
        
        for i in range(8):
            if (landmarks[3*i] < 0) or (landmarks[3*i] >= w) or (landmarks[3*i+1] < 0) or (landmarks[3*i+1] >= h):
                landmarks[3*i] = 0
                landmarks[3*i+1] = 0
                # but it did not matter if they were even negative, only the visibility should be set to zero
                landmarks[3*i+2] = 0
        
        anns[0]['keypoints'] = landmarks
        return image, anns, meta



class ToAnnotations(openpifpaf.transforms.Preprocess):
    def __init__(self, object_annotations):
        self.object_annotations = object_annotations


    def __call__(self, image, anns, meta):
        anns = [
            self.object_annotations[ann['object_type']](**ann)
            for ann in anns
        ]
        return image, anns, meta


def replaceNormalization(compose_transform):
    new_preprocess_list = []
    for op in compose_transform.preprocess_list:
        if isinstance(op, openpifpaf.transforms.NormalizeAnnotations):
            new_preprocess_list.append(NormalizeAnnotations())
        elif isinstance(op, openpifpaf.transforms.Compose):
            new_preprocess_list.append(replaceNormalization(op))
        else:
            new_preprocess_list.append(op)
    return openpifpaf.transforms.Compose(new_preprocess_list)


TRAIN_TRANSFORM = replaceNormalization(openpifpaf.transforms.TRAIN_TRANSFORM)
EVAL_TRANSFORM = replaceNormalization(openpifpaf.transforms.EVAL_TRANSFORM)
