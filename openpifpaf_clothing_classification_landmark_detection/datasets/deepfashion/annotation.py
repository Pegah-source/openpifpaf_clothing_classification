import copy

from .attribute import DeepfashionType, DEEPFASHION_ATTRIBUTE_METAS
from .. import annotation
import numpy as np


class DeepfashionClothingAnnotation(annotation.AnnotationAttr):
    """Annotation class for pedestrians from dataset DEEPFASHION."""
    # base class will take care of adding the function json_data

    object_type = DeepfashionType.Clothing
    attribute_metas = DEEPFASHION_ATTRIBUTE_METAS[DeepfashionType.Clothing]


    def inverse_transform(self, meta):
        pred = copy.deepcopy(self)
        left_top = np.array([meta['original_left'], meta['original_top']]).astype(np.float64)
        left_top = np.array([left_top]*8).reshape((8, 2))
        pred.data += left_top
        return pred


'''DEEPFASHION_OBJECT_ANNOTATIONS = {
    DeepfashionType.Clothing: DeepfashionClothingAnnotation,
}'''
