import openpifpaf

from .basenetwork import ForkNormNetwork
from .decoder import Combined_Dec
from .head import AttributeField
from .loss import CategoryLoss
from ...datasets import headmeta


def register():
    openpifpaf.BASE_TYPES.add(ForkNormNetwork)
    for backbone in list(openpifpaf.BASE_FACTORIES.keys()):
        openpifpaf.BASE_FACTORIES['fn-'+backbone] = (lambda backbone=backbone:
            ForkNormNetwork('fn-'+backbone, backbone))
    openpifpaf.HEADS[headmeta.ClassMeta] = AttributeField
    openpifpaf.DECODERS.add(Combined_Dec)
    openpifpaf.LOSSES[headmeta.ClassMeta] = CategoryLoss
