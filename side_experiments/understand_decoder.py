# in the factory of the decoder:
[   CifCaf([meta], [meta_next])
    for meta, meta_next in zip(head_metas[:-1], head_metas[1:])
    if (isinstance(meta, headmeta.Cif) and isinstance(meta_next, headmeta.Caf))
]

# how the head_metas are added in the datamodule??
self.head_metas = [cif, caf, dcaf] if self.with_dense else [cif, caf]

# in my case I have a self defined head in the beginning and then cif and then caf. So it would be like:
[   CifCaf([meta], [meta_next])
    for meta_class, meta, meta_next in zip(head_metas[0], head_metas[1], head_metas[2])
    if (isinstance(meta, headmeta.ClassMeta) and isinstance(meta, headmeta.Cif) and isinstance(meta_next, headmeta.Caf))
]


# and then you can change the decoder of the cifcaf itself to add also the class annotation.
# why the decoder is used in the predictor??
# the pred_annotation should be of annotation type so that we can call the inverse_transform function.
# but why do we need to call this function?? 
# it's a usage like this:
pred = [ann.inverse_transform(meta) for ann in pred]
gt_anns = [ann.inverse_transform(meta) for ann in gt_anns]
# and annotation is what also has the json_data function
# in the inverse_transform we have some meta['rotation'], where exactly did we define such an attribute?? probably in the pre-processing
meta['rotation']['angle'] = angle
meta['rotation']['width'] = w
meta['rotation']['height'] = h
# we have the inverse_transform so that the output would be comparable to input. In the case of class nothing is needed.

# understand how the decoder is used and what is the input and output and where it will be called in the main code:
# the decoder is used as a way to transform back the predicted features, based on the performed pre-processing, so that
# they would be comparable to the target annotations and keypoints and ....
# how does the decoder work?? the decoder is called over a batch of images and a model. it then calls the __call__ of the decoder itself.
# and output the objects of type (inherited from the type) openpifpaf.annotation.Base. We then call the inverse_transform over these objects
# and output
if self.json_data:
    pred = [ann.json_data() for ann in pred]

yield pred, gt_anns, meta


# so how is it possible to have the annotations and classes all together if their annotation class is not the same??
# otherwise how json_data is going to be used?
# json_data just outputs a dictionary, so they can be appended afterwards?? NO, this way you will have to change the openpifpaf code.


# How does George do it?? 
# you can define a whole new object that has inverse_transform and json_data .... like a combined annotation that inherits from the 