from abc import abstractmethod
from enum import auto, Enum
import logging
from typing import Dict

from .attribute import ObjectType
from .headmeta import ClassMeta


LOG = logging.getLogger(__name__)


class AnnotationRescaler:
    """Rescale images and annotations based on stride of network.

    Args:
        stride (int): Factor to divide dimensions by.
        object_type (ObjectType): Category of object annotated.
    """

    def __init__(self, stride: int, object_type: ObjectType):
        self.stride = stride
        self.object_type = object_type


    def valid_area(self, meta):
        if 'valid_area' not in meta:
            return None

        return (
            meta['valid_area'][0] / self.stride,
            meta['valid_area'][1] / self.stride,
            meta['valid_area'][2] / self.stride,
            meta['valid_area'][3] / self.stride,
        )


    @abstractmethod
    def objects(self, anns):
        """Rescale and return object annotations of given type.
        Needs to be implemented for every object type.
        """
        raise NotImplementedError


    def width_height(self, width_height_original):
        return [round((width_height_original[0]-1) / self.stride + 1),
                round((width_height_original[1]-1) / self.stride + 1)]


class AttributeEncoder:
    """Convert annotations to target feature maps.

    Args:
        meta (AttributeMeta): Description of the attribute.
        rescaler (AnnotationRescaler): Rescaler corresponding to object type.
    """

    def __init__(self,
                 meta: ClassMeta,
                 rescaler: AnnotationRescaler = None,
                 **kwargs):
        self.meta = meta
        self.rescaler = rescaler
        self.__dict__.update(kwargs)


    def __call__(self, image, anns, meta):
        generator = ATTRIBUTE_GENERATORS[self.meta.dataset][self.meta.object_type]
        return generator(self)(image, anns, meta)

# we can define a class that inherits from AttributeGenerator
# and it will implement the generate encoding
class AttributeGenerator:
    """Compute target feature map for an attribute.

    Args:
        config (AttributeEncoder): Meta information about how to handle the
            attribute.
    """

    rescaler_class = AnnotationRescaler


    def __init__(self, config: AttributeEncoder):
        self.config = config
        if config.rescaler is not None or self.rescaler_class is not None:
            self.rescaler = config.rescaler or self.rescaler_class(
                config.meta.stride, config.meta.object_type)
        else:
            self.rescaler = None


    def __call__(self, image, anns, meta):
        # in the case of a single attribue we only need the image_category that is in the meta
       
        if self.rescaler is not None:
            width_height_original = image.shape[2:0:-1]
            objects = self.rescaler.objects(anns)
            new_width_height = self.rescaler.width_height(width_height_original)
            valid_area = self.rescaler.valid_area(meta)
            # add here 
            LOG.debug('valid area: %s', valid_area)
            encoding = self.generate_encoding_preprocessed(objects, new_width_height, valid_area)
        else:
            encoding = self.generate_encoding_from_original(image, anns, meta)

        
        return encoding


    @abstractmethod
    def generate_encoding_preprocessed(self, objects, width_height, valid_area):
        """Compute targets from annotations."""
        raise NotImplementedError
        
    def generate_encoding_from_original(self, image, anns, meta):
        """Compute targets from annotations."""
        raise NotImplementedError


"""List of generatpr for every dataset and object type."""
ATTRIBUTE_GENERATORS: Dict[str, Dict[ObjectType, AttributeGenerator]] = {}
