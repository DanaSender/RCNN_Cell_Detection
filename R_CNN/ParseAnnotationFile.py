# This file is designed to parse the annotation file receives from the labeling tool

# Imports
from PIL import Image
from matplotlib import image
import pickle
import os

WIDTH = 1024
HEIGHT = 1024
CHANNELS = ['Blue', 'Red']
CHANNEL_DICT = {'Blue': 2, 'Red': 0, 'Green': 1}


def create_new_annotation(filename, channel):
    """
    This method creates by the old annotation the new annotation that fits the network
    """
    infile = open(filename, 'rb')
    new_dict = pickle.load(infile)
    print(new_dict)
    infile.close()
    old_boxes = new_dict[channel]
    # extract each bounding box
    boxes = list()
    if len(old_boxes) > 0:
        for box in old_boxes:
            x1, y1 = box[0]
            x2, y2 = box[1]
            xmin = min(x1, x2)
            ymin = min(y1, y2)
            xmax = max(x1, x2)
            ymax = max(y1, y2)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
    return boxes, WIDTH, HEIGHT


def save_new_annotation(results_dir, annots_dir, channels):
    """
    This method saves the new annotations in the wanted path
    """
    for result in os.listdir(results_dir):
        filename = results_dir + result
        for channel in channels:
            boxes, _, _ = create_new_annotation(filename, channel)
            outfile = open(annots_dir + result[:-8] + "_" + channel + "_results", 'wb')
            pickle.dump(boxes, outfile)
            outfile.close()


def image_for_one_channel(orig_image, cnl):
    """
    This method receives an image and a specific channel and returns a black and white image
    only of the requested channel.
    """
    channel_image = orig_image.copy()
    channel_image[:, :, 0] = channel_image[:, :, cnl]
    channel_image[:, :, 1] = channel_image[:, :, cnl]
    channel_image[:, :, 2] = channel_image[:, :, cnl]
    return channel_image


def create_new_images(old_image_dir, new_images_dir, channels):
    """
    This method creates by the old image the new image according to the receives channels
    """
    for img_name in os.listdir(old_image_dir):
        filename = old_image_dir + img_name
        img = image.imread(filename)
        for channel in channels:
            ##### next line only if we want black-white image
            one_channel_img = image_for_one_channel(img, CHANNEL_DICT[channel])
            im = Image.fromarray(one_channel_img)
            im.save(new_images_dir + img_name[:-4] + "_" + channel + ".tif")


########################################################################
# prepare annotation files
# make from one annot two anoots of blue and red channel
results_dir = 'G:/Dana/Mask R-CNN/cell_data_new/Butrus/results/'
annots_dir = 'G:/Dana/Mask R-CNN/cell_data_new/Butrus/annots/'
# copy to annots folder the result file with the same name and _blue in the end of the nam3
save_new_annotation(results_dir, annots_dir, ['Green', "Red"])

# prepare image files
# make from one image two images of blue and red channel (black-white) images
old_image_dir = 'G:/Dana/Mask R-CNN/cell_data_new/Butrus/old_images/'
new_images_dir = 'G:/Dana/Mask R-CNN/cell_data_new/Butrus/images/'
create_new_images(old_image_dir, new_images_dir, ['Green', "Red"])
