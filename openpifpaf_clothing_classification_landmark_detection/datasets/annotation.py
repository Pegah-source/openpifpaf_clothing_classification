from abc import abstractmethod
from typing import Dict

import openpifpaf
import copy
from .attribute import ObjectType



class AnnotationAttr(openpifpaf.annotation.Base):
    """Annotation class for a detected instance."""

    object_type = None
    attribute_metas = None


    def __init__(self, **kwargs):
        self.attributes = {}
        for meta in self.attribute_metas: # meta is the whole list of the attribute, group, n_channels and ....
            if meta['attribute'] in kwargs:
                self.attributes[meta['attribute']] = kwargs[meta['attribute']]

    def json_data(self):
        return {'object_type': self.object_type.name, **self.attributes}


"""List of annotations for every dataset and object type."""
# OBJECT_ANNOTATIONS: Dict[str, Dict[ObjectType, AnnotationAttr]] = {}
OBJECT_ANNOTATIONS: Dict[str, Dict[ObjectType, openpifpaf.annotation.Base]] = {}