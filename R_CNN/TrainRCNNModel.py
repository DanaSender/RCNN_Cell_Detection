# This file fits a R-CNN on the cell dataset

# Imports
from os import listdir
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import pickle

NUM_OF_TRAIN_IMAGES = 710
NUM_OF_TOTAL_IMAGES = 710 + 37


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


# This class defines a configuration for the model
class cellConfig(Config):
    # define the name of the configuration
    NAME = "cell_cfg"
    # number of classes (background + cell)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = NUM_OF_TRAIN_IMAGES


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
# prepare config
config = cellConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
