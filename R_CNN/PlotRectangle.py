# detect cells in photos with mask rcnn model
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

WIDTH = 1024
HEIGHT = 1024
CHANNEL_DICT = {'Blue': 2, 'Red': 0, 'Green': 1}


# load old images and annots
def plot_old_boxes(filename, channel):
    infile = open(filename, 'rb')
    new_dict = pickle.load(infile)
    infile.close()
    ax = pyplot.gca()
    old_boxes = new_dict[channel]
    for box in old_boxes:
        # get coordinates
        x1, y1 = box[0]
        x2, y2 = box[1]
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
    pyplot.show()


# load new images and annots
def plot_new_boxes(filename):
    infile = open(filename, 'rb')
    boxes = pickle.load(infile)
    infile.close()
    ax = pyplot.gca()
    for box in boxes:
        # get coordinates
        x1, y1, x2, y2 = box
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
    pyplot.show()


# make black-white images
def image_for_one_channel(orig_image, cnl):
    channel_image = orig_image.copy()
    channel_image[:, :, 0] = channel_image[:, :, cnl]
    channel_image[:, :, 1] = channel_image[:, :, cnl]
    channel_image[:, :, 2] = channel_image[:, :, cnl]
    return channel_image


def plot_image(image_name, channel=None):
    image = mpimg.imread(image_name)
    if channel is not None:
        image = image_for_one_channel(image, CHANNEL_DICT[channel])
    pyplot.imshow(image)


# Plot old boxes (of old annots)
image_name = 'G:/Dana/Mask R-CNN/cell_data_DR_B_I_B_A/test_set/images/562.tif'
annots_name = 'G:/Dana/Mask R-CNN/cell_data_DR_B_I_B_A/test_set/annots/562'
plot_image(image_name, "Red")
plot_old_boxes(annots_name, "Red")


# Plot new boxes (of new annots)
# image_name = 'G:/Dana/Mask R-CNN/cell_data/images/562.tif'
# annots_name = 'G:/Dana/Mask R-CNN/cell_data/annots/562'
# plot_image(image_name, "Blue")
# plot_new_boxes(annots_name)




