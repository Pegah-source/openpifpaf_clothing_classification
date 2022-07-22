import openpifpaf

from .combinedAnnotation import DEEPFASHION_OBJECT_ANNOTATIONS
from .attribute import DeepfashionType, DEEPFASHION_ATTRIBUTE_METAS
from .datamodule import Deepfashion_module
from .encoder import DEEPFASHION_ATTRIBUTE_GENERATORS
from .. import annotation
from .. import attribute
from .. import encoder
from .. import painter


def register():
    print('registration of deepfashion module')
    openpifpaf.DATAMODULES['deepfashion'] = Deepfashion_module
    # openpifpaf.PAINTERS['JaadPedestrianAnnotation'] = painter.BoxPainter

    attribute.OBJECT_TYPES['deepfashion'] = DeepfashionType
    attribute.ATTRIBUTE_METAS['deepfashion'] = DEEPFASHION_ATTRIBUTE_METAS
    encoder.ATTRIBUTE_GENERATORS['deepfashion'] = DEEPFASHION_ATTRIBUTE_GENERATORS
    annotation.OBJECT_ANNOTATIONS['deepfashion'] = DEEPFASHION_OBJECT_ANNOTATIONS # transforms should be modified now
    # to be compatible with combinedAnnotation
