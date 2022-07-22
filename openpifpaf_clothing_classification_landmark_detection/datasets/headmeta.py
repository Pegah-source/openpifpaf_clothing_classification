from dataclasses import dataclass
from typing import Dict, List, Union

import openpifpaf

from .attribute import ObjectType


@dataclass
class ClassMeta(openpifpaf.headmeta.Base):
    """Meta information about an attribute.

    Args:
        object_type (ObjectType): Type of object annotated.
        category (str): Name of category.
        labels (Dict[int, str]): Names of classes.
    """

    object_type: ObjectType
    attribute: str
    n_channels : int
    group: str # classification
    default: Union[int, float, List[float]] = None
    labels: Dict[int, str] = None

    @property
    def n_fields(self):
        return 1