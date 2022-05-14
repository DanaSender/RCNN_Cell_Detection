# This software uses weights trained to predicate new images.
# In order for the software to work, it needs to receive a path to a folder that has the
# images on which we want to do the prediction.

from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from matplotlib import image
import argparse
import os

CHANNEL_DICT = {'Blue': 2, 'Red': 0, 'Green': 1}


def read_files(folder):
    # in case the folder does not exists
    images_paths = []
    for filename in os.listdir(folder):
        if filename.endswith('.tif'):
            images_paths.append(folder + "/" + filename)
    return images_paths


# make black-white images
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


def plot_boxes_on_image(axis, fig, i, yhat):
    """
    This method plots bounding boxes of all the cells found on the image
    """
    for box in yhat['rois']:
        # get coordinates
        y1, x1, y2, x2 = box
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        # draw the box
        axis[i].add_patch(rect)
    # show the figure
    w = 15
    h = 10
    fig.set_size_inches(w, h)


def plot_predicted(model, image_path, cfg, count_dict):
    """
    This method receives a model and a path of an image and plots for each channel the prediction of the model
    and the number of cells found
    """
    fig, axis = pyplot.subplots(1, 3)
    for i, channel in enumerate(CHANNEL_DICT):
        img = image.imread(image_path)
        img = image_for_one_channel(img, CHANNEL_DICT[channel])
        scaled_image = mold_image(img, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        num_of_rect = len(yhat['rois'])
        count_dict[channel] += num_of_rect
        # plot raw pixel data
        axis[i].imshow(img)
        axis[i].set_title(channel + " channel: "+ str(num_of_rect) +" cells")
        axis[i].axis('off')
        plot_boxes_on_image(axis, fig, i, yhat)
    pyplot.show()
    return count_dict


# This class define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "cell_cfg"
    # number of classes (background + cell)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def run_program(images_path):
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    model_path = '../R_CNN/rcnn_cell_weights.h5'
    model.load_weights(model_path, by_name=True)
    count_dict = {'Blue': 0, 'Red': 0, 'Green': 0}
    for img in images_path:
        count_dict = plot_predicted(model, img, cfg, count_dict)
    print("Total cells in all images: " + "\nBlue channel: " + str(count_dict["Blue"])+"\nRed channel: " +
          str(count_dict["Red"])+"\nGreen channel: "+str(count_dict["Green"]))


if __name__ == '__main__':
    # read the args - folder always
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True, help="Path to the folder")
    ap.add_argument("-s", "--save", required=False, help="Path to the result")
    args = vars(ap.parse_args())

    # Checks whether the user entered a folder path
    if args["folder"] is not None:
        images_paths = read_files(args["folder"])
        run_program(images_paths)
    else:
        print("You should enter a folder name")