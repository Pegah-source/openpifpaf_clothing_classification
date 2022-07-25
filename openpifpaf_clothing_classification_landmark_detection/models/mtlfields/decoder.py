import argparse
import logging
import time
from typing import List

import numpy as np
import torch

from ...datasets import headmeta
import openpifpaf
from .class_decoder import ClassDecoder
from ...datasets import attribute
from ...datasets.deepfashion.combinedAnnotation import AnnotationCombined


LOG = logging.getLogger(__name__)


class Combined_Dec(openpifpaf.decoder.decoder.Decoder):
    
    def __init__(self,
                 class_metas : List[headmeta.ClassMeta],
                 cif_metas: List[openpifpaf.headmeta.Cif],
                 caf_metas: List[openpifpaf.headmeta.Caf],
                 *,
                 cif_visualizers=None,
                 cifhr_visualizers=None,
                 caf_visualizers=None):
        super().__init__()
        '''
            **************
            start of the initialization of the CifCaf decoder
            **************
        '''
        cifcaf_decoder = openpifpaf.decoder.cifcaf.CifCaf(cif_metas, caf_metas, cif_visualizers=cif_visualizers, cifhr_visualizers=cifhr_visualizers, caf_visualizers=caf_visualizers)
        self.cifcaf_decoder = cifcaf_decoder
        '''
            **************
            end of the initialization of the CifCaf decoder
            **************
        '''
        
        '''
            $$$$$$$$$$$$$$$
            start of the initialization of the class decoder
            $$$$$$$$$$$$$$$
        '''
        for dataset in attribute.OBJECT_TYPES:
            for object_type in attribute.OBJECT_TYPES[dataset]:
                meta_list = [meta for meta in class_metas
                             if (
                                isinstance(meta, headmeta.ClassMeta)
                                and (meta.dataset == dataset)
                                and (meta.object_type is object_type)
                             )]
                class_decoder = ClassDecoder(dataset=dataset, object_type=object_type, attribute_metas=meta_list)
        self.class_decoder = class_decoder

        '''
            $$$$$$$$$$$$$$$
            end of the initialization of the class decoder
            $$$$$$$$$$$$$$$
        '''

    

    @classmethod
    def factory(cls, head_metas):
    
        return [
            Combined_Dec([meta_class], [meta], [meta_next])
            for meta_class, meta, meta_next in zip(head_metas[1], head_metas[1:-1], head_metas[2:])
            if (isinstance(meta, openpifpaf.headmeta.Cif)
                and isinstance(meta_next, openpifpaf.headmeta.Caf)
                and isinstance(meta_class, headmeta.ClassMeta))
        ]

    def __call__(self, fields, initial_annotations=None):


        cifcaf_annotations = self.cifcaf_decoder(fields, initial_annotations=initial_annotations) # of type openpifpaf.annotation.Annotation
        class_annotations = self.class_decoder(fields, initial_annotations=initial_annotations)# of type annotation.DeepfashionClothingAnnotation

        # should return an ensemble of these two annotations
        combined_annotation = []
        for cifcaf_anno, class_anno in zip(cifcaf_annotations, class_annotations):
            combined_annotation.append(AnnotationCombined(class_anno, cifcaf_anno))
         

        LOG.info('annotations %d: %s',
                 len(combined_annotation),
                 [np.sum(ann.cifcaf_annotation.data[:, 2] > 0.1) for ann in combined_annotation])

        return combined_annotation