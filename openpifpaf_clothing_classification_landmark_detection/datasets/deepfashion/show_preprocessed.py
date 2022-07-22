# the goal is to show the preprocessed images and see if everything is consistent
from .train_loader_test import Deepfashion_module
from calendar import EPOCH
import pandas as pd
import torch.utils.data
from PIL import Image
import numpy as np
# importing libraries

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from contextlib import contextmanager
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import argparse
import os

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

DEEPFASHION_SKELETON = [
    [1, 2], # left and right collar connection
    [1, 3],
    [1, 5], # left hem and left sleeve connection
    [5, 6], # left and right hem connection I AM NOT SURE IF THIS SHOULD BE IN THE SKELETON
    [6, 2], # right sleeve and right collar connection
    [4, 2],
    [5, 7], 
    [7, 8],
    [8, 6],
]
# load 20 random images from the dataset and show them along with their normalized error for each in pic landmark.

            
def kp_visualizer(temp_image_name, np_image, gt_lms, gt_visib):

    current_path = os.getcwd()
    images_path = os.path.join(current_path, 'images')
    images_exist = os.path.exists(images_path)

    if not images_exist:
        os.makedirs(images_path)
        print("images directory is created!")

    with image_canvas(np_image, show=False, fig_file= 'images/'+temp_image_name + '.png', fig_width=10, dpi_factor=1.0) as ax:

        keypoints(ax, gt_visib, gt_lms, 'bo', size=np_image.size)


    return

def keypoints(ax, vis, lms, differentizer_color,  individual_pic_sample_dist=None, total_score=None,  *,
              size=None, scores=None, color=None,
              colors=None, texts=None, activities=None, dic_out=None):
    
    
    if lms is None:
        return

    if color is None and colors is None:
        colors = range(len(lms))

    # we only have one set of keypoints here
    for i, (visib, lm) in enumerate(zip(np.asarray(vis), np.asarray(lms))):
        assert lm.shape[1] == 2
        # assert len(vis) == 1
        xy_scale = 1
        y_scale = 1
        x = lm[:, 0] * xy_scale
        y = lm[:, 1] * xy_scale * y_scale
        v = visib[:, 0] # visibility

       
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)
        if total_score is not None:
            ax.text(450 + 2, 500 - 2, total_score, fontsize=20, color='red', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0})
        _draw_skeleton(ax, x, y, v, differentizer_color, individual_pic_sample_dist,  i=i, size=size, color=color, activities=activities, dic_out=dic_out)

        if texts is not None:
            draw_text(ax, x, y, v, texts[i], color)

       
def _draw_skeleton(ax, x, y, v, differentizer_color, individual_pic_sample_dist=None,  *, i=0, size=None, color=None, activities=None, dic_out=None):
        
    if not np.any(v > 0):
        return
    for n_joint, (joint_x, joint_y) in enumerate(zip(x, y)):
        if v[n_joint] > 0:
            c = color
            ax.plot(joint_x, joint_y, differentizer_color)
            if individual_pic_sample_dist is not None:
                ax.text(joint_x + 2, joint_y - 2, str(individual_pic_sample_dist[n_joint]), fontsize=18, color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0})

    if DEEPFASHION_SKELETON is not None:
        for ci, connection in enumerate(np.array(DEEPFASHION_SKELETON) - 1):
            # cnnection is of form [a, b]
            c = color
            linewidth = 2

            color_connections = False
            dashed_threshold = 0.005
            solid_threshold = 0.005
            if color_connections:
                c = matplotlib.cm.get_cmap('tab20')(ci / len(DEEPFASHION_SKELETON))
            if np.all(v[connection] > dashed_threshold):
                print('visibs of a connection ', v[connection])
                ax.plot(x[connection], y[connection],
                        linewidth=linewidth, color=c,
                        linestyle='dashed', dash_capstyle='round')
            if np.all(v[connection] > solid_threshold):
                print('visibs of a connection 1 ', v[connection])
                ax.plot(x[connection], y[connection],
                        linewidth=linewidth, color=c, solid_capstyle='round')
        


  
@contextmanager
def image_canvas(image, fig_file=None, show=True, dpi_factor=1.0, fig_width=10.0, **kwargs):
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (fig_width, fig_width*image.shape[0]/image.shape[1])

    if ndimage is None:
        raise Exception('please install scipy')
    fig = plt.figure(**kwargs)
    canvas = FigureCanvas(fig)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    fig.add_axes(ax)
    # image_2 = ndimage.gaussian_filter(image, sigma=2.5)
    ax.imshow(image, alpha=0.4)
    yield ax

    canvas.draw()       # draw the canvas, cache the renderer
    imaged_plot = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    # this image should be sent to running code part to be added to tensorboard
    if fig_file:
        fig.savefig(fig_file, dpi=image.shape[0] / kwargs['figsize'][0] * dpi_factor)
        print('keypoints image saved')

    
    plt.close(fig)


def draw_text(ax, x, y, v, text, color, fontsize=8):
    if not np.any(v > 0):
        return

    # keypoint bounding box
    x1, x2 = np.min(x[v > 0]), np.max(x[v > 0])
    y1, y2 = np.min(y[v > 0]), np.max(y[v > 0])
    if x2 - x1 < 5.0:
        x1 -= 2.0
        x2 += 2.0
    if y2 - y1 < 5.0:
        y1 -= 2.0
        y2 += 2.0

    ax.text(x1 + 2, y1 - 2, text, fontsize=fontsize,
            color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0})


if __name__ == "__main__":
    deepfashion_module = Deepfashion_module()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1, type = int, help='batch size')
    parser.add_argument('--debug', default=False, action='store_true', help='print debug messages')

    deepfashion_module.cli(parser)
    args = parser.parse_args()
    args.pin_memory = False
    deepfashion_module.configure(args)
    training_data = deepfashion_module.train_loader()


    dataloader_iterator = iter(training_data)
    data, target, meta = next(dataloader_iterator)
    print('just to make sure ' , data.shape, '  ', type(target), '  ', target)
    target = target[0]
    data = data[0]
    meta = meta[0]
    gt_lms_and_visibs = target['keypoints'][0]

    gt_lms = np.array([gt_lms_and_visibs[i,j] for i in range(8) for j in range(2)]).reshape((1, 8, 2))
    gt_visib = np.array([gt_lms_and_visibs[i,2] for i in range(8)]).reshape((1, 8, 1))
    data = np.transpose(data, (1, 2, 0))
    # first standardize the data
    mean_data = torch.mean(data)
    std_data = torch.std(data)
    data = (data - mean_data)/std_data
    print('here is the min ', torch.min(data))
    data = ((data-torch.min(data))*255.0)/(torch.max(data)-torch.min(data))
    
    data = data.int()
    print('here is the data ', data)
    plt.imshow(data)
    plt.show()
    kp_visualizer('pejjahi'+'_'+str(1), data, gt_lms, gt_visib)