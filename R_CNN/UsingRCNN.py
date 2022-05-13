# detect cells in photos with mask rcnn model
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
    channel_image = orig_image.copy()
    channel_image[:, :, 0] = channel_image[:, :, cnl]
    channel_image[:, :, 1] = channel_image[:, :, cnl]
    channel_image[:, :, 2] = channel_image[:, :, cnl]
    return channel_image


# def plot_predicted(model, image_path, channel, cfg):
#     fig = pyplot.figure()
#     pyplot.title(image_path)
#     pyplot.axis('off')
#     img = image.imread(image_path)
#     img = image_for_one_channel(img, CHANNEL_DICT[channel])
#     scaled_image = mold_image(img, cfg)
#     # convert image into one sample
#     sample = expand_dims(scaled_image, 0)
#     # make prediction
#     yhat = model.detect(sample, verbose=0)[0]
#     # plot raw pixel data
#     pyplot.imshow(img)
#     # pyplot.title(image_path + "  channel " + channel)
#     pyplot.title("Predicted")
#     ax = pyplot.gca()
#     ax.axis('off')
#     # plot each box
#     for box in yhat['rois']:
#         # get coordinates
#         y1, x1, y2, x2 = box
#         # calculate width and height of the box
#         width, height = x2 - x1, y2 - y1
#         # create the shape
#         rect = Rectangle((x1, y1), width, height, fill=False, color='red')
#         # draw the box
#         ax.add_patch(rect)
#     # show the figure
#     w = 15
#     h = 10
#     fig.set_size_inches(w, h)
#     # pyplot.savefig('cell_cfg20210706T1814/compared_images/image_id_'+str(i)+'_'+data_set_name)
#     pyplot.show(block=True)


def count_cells_in_image(model, image_path, cfg, count_dict):
    for channel in CHANNEL_DICT:
        img = image.imread(image_path)
        img = image_for_one_channel(img, CHANNEL_DICT[channel])
        scaled_image = mold_image(img, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        num_of_rect = len(yhat['rois'])
        # print(image_path + "\n" + channel + " num of tags: " + str(num_of_rect))
        count_dict[channel] += num_of_rect
        print(image_path)
    return count_dict


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "cell_cfg"
    # number of classes (background + cell)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def run_program(images_paths):
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    model_path = 'cell_cfg20220128T0949_743_DRBA/mask_rcnn_cell_cfg_0005.h5'  # TODO
    # model_path = 'cell_cfg_first_try/mask_rcnn_cell_cfg_0005.h5'
    model.load_weights(model_path, by_name=True)
    count_dict = {'Blue': 0, 'Red': 0, 'Green': 0}
    for image_path in images_paths:
        # plot_predicted(model, image_path, "Blue", cfg)
        # plot_predicted(model, image_path, "Red", cfg)
        count_dict = count_cells_in_image(model, image_path, cfg, count_dict)
    print(count_dict)


if __name__ == '__main__':
    # read the args - folder always
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True, help="Path to the folder")
    ap.add_argument("-s", "--save", required=False, help="Path to the result")
    args = vars(ap.parse_args())

    # check if specific image was entered (if yes- run only the image, if no- go over all the folder)
    if args["folder"] is not None:
        images_paths = read_files(args["folder"])
        run_program(images_paths)
    else:
        print("You should enter a folder name")

    # images_paths = read_files("G:/Dana/Images for counting/images_TPH1/pat29stomach1")
    # run_program(images_paths)
