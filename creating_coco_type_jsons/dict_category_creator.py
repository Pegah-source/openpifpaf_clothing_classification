from nis import cat
from  constants import DEEPFASHION_KEYPOINTS, DEEPFASHION_SKELETON

class MyStr(str):
    """ Special string subclass to override the default representation method
        which puts single quotes around the result.
    """
    def __repr__(self):
        return super(MyStr, self).__repr__().strip("'")


category_root_file = '/Users/pegahkhayatan/Desktop/classification_pifpaf/creating_coco_type_jsons/'
categories_file = category_root_file + 'list_category_cloth.txt'
categories = open(categories_file).readlines()[2:]

all_dict_category = []
dict_type_category = {1:'upper', 2:'lower', 3:'full'}
for index, line in enumerate(categories):
    dict_category = {}
    info = line.split()
    cat_name = info[0]
    cat_number = info[1]
    dict_category["name"] = cat_name
    dict_category["id"] = index
    dict_category["skeleton"] = MyStr('list(DEEPFASHION_SKELETON)')
    dict_category["supercategory"] = dict_type_category[int(cat_number)]
    dict_category["keypoints"] = MyStr('list(DEEPFASHION_KEYPOINTS)')

    all_dict_category.append(dict_category)

'''
    [{  "name":'clothing',
        "id":1,
        "skeleton":list(DEEPFASHION_SKELETON),
        "supercategory":'full',
        "keypoints":list(DEEPFASHION_KEYPOINTS)}]
'''

for dict in all_dict_category:
    print(dict, ',')

