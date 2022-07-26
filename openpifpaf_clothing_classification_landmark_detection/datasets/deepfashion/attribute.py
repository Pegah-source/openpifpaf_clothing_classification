from ..attribute import ObjectType


class DeepfashionType(ObjectType):
    """Object types for deepfashion dataset."""
    Clothing = ()


DEEPFASHION_ATTRIBUTE_METAS = {
    DeepfashionType.Clothing: [
        {'attribute': 'class', 'group': 'classification', 'n_channels': 50, 'default': 2, 'labels': {0: 'Anorak' ,
                                                                                                    1: 'Blazer' ,
                                                                                                    2: 'Blouse' ,
                                                                                                    3: 'Bomber' ,
                                                                                                    4: 'Button-Down' ,
                                                                                                    5: 'Cardigan' ,
                                                                                                    6: 'Flannel' ,
                                                                                                    7: 'Halter' ,
                                                                                                    8: 'Henley' ,
                                                                                                    9: 'Hoodie' ,
                                                                                                    10: 'Jacket' ,
                                                                                                    11: 'Jersey' ,
                                                                                                    12: 'Parka' ,
                                                                                                    13: 'Peacoat' ,
                                                                                                    14: 'Poncho' ,
                                                                                                    15: 'Sweater' ,
                                                                                                    16: 'Tank' ,
                                                                                                    17: 'Tee' ,
                                                                                                    18: 'Top' ,
                                                                                                    19: 'Turtleneck' ,
                                                                                                    20: 'Capris' ,
                                                                                                    21: 'Chinos' ,
                                                                                                    22: 'Culottes' ,
                                                                                                    23: 'Cutoffs' ,
                                                                                                    24: 'Gauchos' ,
                                                                                                    25: 'Jeans' ,
                                                                                                    26: 'Jeggings' ,
                                                                                                    27: 'Jodhpurs' ,
                                                                                                    28: 'Joggers' ,
                                                                                                    29: 'Leggings' ,
                                                                                                    30: 'Sarong' ,
                                                                                                    31: 'Shorts' ,
                                                                                                    32: 'Skirt' ,
                                                                                                    33: 'Sweatpants' ,
                                                                                                    34: 'Sweatshorts' ,
                                                                                                    35: 'Trunks' ,
                                                                                                    36: 'Caftan' ,
                                                                                                    37: 'Cape' ,
                                                                                                    38: 'Coat' ,
                                                                                                    39: 'Coverup' ,
                                                                                                    40: 'Dress' ,
                                                                                                    41: 'Jumpsuit' ,
                                                                                                    42: 'Kaftan' ,
                                                                                                    43: 'Kimono' ,
                                                                                                    44: 'Nightdress' ,
                                                                                                    45: 'Onesie' ,
                                                                                                    46: 'Robe' ,
                                                                                                    47: 'Romper' ,
                                                                                                    48: 'Shirtdress' ,
                                                                                                    49: 'Sundress' ,}}
                                                                                                        ]}
