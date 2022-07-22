import copy

import numpy as np
from openpifpaf.utils import mask_valid_area
import torch

from .encoder import AnnotationRescaler, AttributeGenerator


class SingleAttributeGenerator(AttributeGenerator):
    """AttributeGenerator for objects defined with bounding boxes."""
    
    rescaler_class = None


    def generate_encoding_from_original(self, image, anns, meta):

        n_targets = self.config.meta.n_channels
        self.targets = np.full(
            (n_targets),
            0,
            dtype=np.float32,
        )

        self.targets[int(meta['image_category'])] = 1
        
        # return torch.from_numpy(meta['image_category'])
        return torch.from_numpy(self.targets)