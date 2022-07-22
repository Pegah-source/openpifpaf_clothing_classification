from .attribute import DeepfashionType
from .. import generators


DEEPFASHION_ATTRIBUTE_GENERATORS = {
    DeepfashionType.Clothing: generators.SingleAttributeGenerator,
}
