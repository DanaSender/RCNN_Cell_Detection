# This file is designed to detect cells in new images (wo actual labeling) with R-CNN model

# Imports
from os import listdir
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from matplotlib import image
import pickle

NUM_OF_TRAIN_IMAGES = 710
NUM_OF_TOTAL_IMAGES = 710 + 37
CHANNEL_DICT = {'Blue': 2, 'Red': 0, 'Green': 1}


# This class defines and loads the cell dataset
class cellDataset(Dataset):
    def load_dataset(self, dataset_dir):
        """
        This method receives directory and loads the dataset
        """
        from random import shuffle
        # define one class
        self.add_class("dataset", 1, "cell")
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        files = listdir(images_dir)
        shuffle(files)
        # find all images
        for filename in files:
            image_id = filename[:-4]
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + "_results"
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_boxes(self, image_id):
        """
        This method receives an image id and loads the ground truth
        """
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        infile = open(path, 'rb')
        boxes = pickle.load(infile)
        infile.close()
        return boxes

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# This class define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "cell_cfg"
    # number of classes (background + cell)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


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


def plot_predicted(model, image_path, channel):
    """
    This method receives a model, path of an image and a channel and plots on the image (with the relevant channel)
    the prediction obtained from the model
    """
    fig = pyplot.figure()
    pyplot.title(image_path)
    pyplot.axis('off')
    img = image.imread(image_path)
    img = image_for_one_channel(img, CHANNEL_DICT[channel])
    scaled_image = mold_image(img, cfg)
    # convert image into one sample
    sample = expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)[0]
    # plot raw pixel data
    pyplot.imshow(img)
    # pyplot.title(image_path + "  channel " + channel)
    pyplot.title("Predicted")
    ax = pyplot.gca()
    ax.axis('off')
    # plot each box
    for box in yhat['rois']:
        # get coordinates
        y1, x1, y2, x2 = box
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
    # show the figure
    w = 15
    h = 10
    fig.set_size_inches(w, h)
    pyplot.show(block=True)


# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(data_set_name, dataset, model, cfg, n_images=10):
    """
    This method plots for n_images both the ground truth and the prediction obtained from the model
    """
    for i in range(n_images):
        print(i)
        fig = pyplot.figure()
        pyplot.title(data_set_name + ' dataset')
        pyplot.axis('off')
        image = dataset.load_image(i)
        boxes = dataset.load_boxes(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        fig.add_subplot(1, 2, 1)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Actual')
        ax = pyplot.gca()
        ax.axis('off')
        # plot each box
        for box in boxes:
            # get coordinates
            x1, y1, x2, y2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
        fig.add_subplot(1, 2, 2)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Predicted')
        ax = pyplot.gca()
        ax.axis('off')
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
        # show the figure
        w = 15
        h = 10
        fig.set_size_inches(w, h)
        pyplot.show(block=True)


if __name__ == '__main__':
    # prepare train set
    train_set = cellDataset()
    train_set.load_dataset('cell_data_DRBA/train_set')
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    # prepare test/val set
    test_set = cellDataset()
    test_set.load_dataset('cell_data_DRBA/test_set')
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    model_path = 'rcnn_cell_weights.h5'
    # model_path = 'cell_cfg_first_try/mask_rcnn_cell_cfg_0005.h5'
    model.load_weights(model_path, by_name=True)
    # plot predictions for train dataset
    # plot_actual_vs_predicted2("train", train_set, model, cfg,['pat2 body03 cy2HDC cy35HT cy5CHGA_SYP32'])
    image_path = 'G:/Dana/Mask R-CNN/all_cells_images/Amit/pat30/images_already_finish/pt30_cy2SYP_cy3CLPS_cy55HT29.tif'
    # image_path = "G:/Dana/Mask R-CNN/cell_data_DR_B_I_B_A/test_set/images/563.tif"
    plot_predicted(model, image_path, "Green")
    exit()
    plot_actual_vs_predicted("train", train_set, model, cfg, 10)
    # plot predictions for test dataset0
    plot_actual_vs_predicted("test", test_set, model, cfg, 10)
