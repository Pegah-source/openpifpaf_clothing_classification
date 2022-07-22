from .annotation import DeepfashionClothingAnnotation
from .attribute import DeepfashionType
import openpifpaf




class AnnotationCombined(openpifpaf.annotation.Base):
    """Anotation class that combines the Annotation class from openpifpaf and DeepfashionClothingAnnotation from deepfashion.annotation"""
    def __init__(self, class_annot : DeepfashionClothingAnnotation, cifcaf_annot : openpifpaf.annotation.Annotation):
        self.class_annotation = class_annot
        self.cifcaf_annotation = cifcaf_annot
        
    def inverse_transform(self, meta):
        ann_cifcaf = self.cifcaf_annotation.inverse_transform(meta) # of type Annotation
        ann_class = self.class_annotation.inverse_transform(meta) # of type DeepfashionClothingAnnotation
        '''left_top = np.array([meta['original_left'], meta['original_top']])
            print('left_top size ', left_top.shape)
            left_top = np.array([left_top]*8).reshape((8, 2))
            non_normalized_output[batch_n] += left_top
            non_normalized_sample_landmarks[batch_n] += left_top'''
        # I don't know where to add the inverse transform for cropping the bounding box
        new_combined_ann = AnnotationCombined(ann_class, ann_cifcaf)
        return new_combined_ann
        

    def json_data(self, coordinate_digits=2):

        # merging the data output of the two annotations:
        class_data = self.class_annotation.json_data(coordinate_digits=coordinate_digits)
        cifcaf_data = self.cifcaf_annotation.json_data(coordinate_digits=coordinate_digits)
        whole_data = {**class_data, **cifcaf_data}

        return whole_data


DEEPFASHION_OBJECT_ANNOTATIONS = {
    DeepfashionType.Clothing: AnnotationCombined,
}
