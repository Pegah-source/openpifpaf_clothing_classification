from . import deepfashion


def register():
    print('registering the dataset modules')
    deepfashion.register()
