import copy

from .attribute import DeepfashionType, DEEPFASHION_ATTRIBUTE_METAS
from .. import annotation


class DeepfashionClothingAnnotation(annotation.AnnotationAttr):
    """Annotation class for pedestrians from dataset DEEPFASHION."""
    # base class will take care of adding the function json_data

    object_type = DeepfashionType.Clothing
    attribute_metas = DEEPFASHION_ATTRIBUTE_METAS[DeepfashionType.Clothing]


    def inverse_transform(self, meta):
        pred = copy.deepcopy(self)
        # no matter what pre-processing we perform, the class remains the same!!(what?? :D )
        # we should inverse the bbox cropping here, after inversing the other
        return pred


'''DEEPFASHION_OBJECT_ANNOTATIONS = {
    DeepfashionType.Clothing: DeepfashionClothingAnnotation,
}'''
