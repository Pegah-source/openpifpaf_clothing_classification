from PIL import Image
from skimage import io
from matplotlib import pyplot as plt
import numpy as np

local_file_path = '/Users/pegahkhayatan/Desktop/deepfashion_project/dataset_inf/img/Zippered_Crew_Neck_Sweater/img_00000056.jpg'
with open(local_file_path, 'rb') as f:
    image = Image.open(f).convert('RGB')

image_2 = io.imread(local_file_path)
print('type of the image ', type(image))
print('method used by Taylor : ', image.size)
print('method used in the other preprocessings : ', image_2.shape)
image = np.array(image)
print('one pixel of the image ', image[0,0])


plt.figure(figsize=(8, 12))
plt.subplot(1, 2, 1), plt.imshow(image), plt.ylabel('PNG loaded with PIL')
plt.subplot(1, 2, 2), plt.imshow(image_2), plt.ylabel('PNG loaded with skimage')
plt.show()

'''
method used by Taylor :  (199, 301)
method used in the other preprocessings :  (301, 199)

so the difference is what method we use to input the image, but then the size or shape should be chosen
accordingly.
what is the difference between these two functions??

'''