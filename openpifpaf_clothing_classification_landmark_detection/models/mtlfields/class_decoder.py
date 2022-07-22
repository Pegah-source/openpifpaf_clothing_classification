import argparse
from dataclasses import field
import logging
import time
from typing import List

import numpy as np
import openpifpaf
import torch

from ...datasets import annotation
from ...datasets import attribute
from ...datasets import headmeta


LOG = logging.getLogger(__name__)


class ClassDecoder(openpifpaf.decoder.decoder.Decoder):
    """Decoder to convert predicted fields to sets of instance detections.

    Args:
        dataset (str): Dataset name.
        object_type (ObjectType): Type of object detected.
        attribute_metas (List[AttributeMeta]): List of meta information about
            predicted attributes.
    """

    # General
    dataset = None
    object_type = None
   


    def __init__(self,
                 dataset: str,
                 object_type: attribute.ObjectType,
                 attribute_metas: List[headmeta.ClassMeta]):
        super().__init__()
        self.dataset = dataset
        self.object_type = object_type
        self.annotation = annotation.OBJECT_ANNOTATIONS[self.dataset][self.object_type]
        for meta in attribute_metas:
            assert meta.dataset == self.dataset
            assert meta.object_type is self.object_type
        self.attribute_metas = attribute_metas



    def __call__(self, fields, initial_annotations=None):
        # problem: which field should be used from all the fields?? so we can create the DeepfashionClothingAnnotation
        start = time.perf_counter()
        fields = [f.numpy() if torch.is_tensor(f) else f for f in fields]
        fields_prob = fields[self.attribute_metas[0].head_index].copy()
        predictions = []
        attributes = {}
        attributes[self.attribute_metas[0].attribute] = fields_prob
        pred = self.annotation(**attributes)
        predictions.append(pred)

        LOG.info('predictions: %d, %.3fs',
                  pred, time.perf_counter()-start)

        return pred        
