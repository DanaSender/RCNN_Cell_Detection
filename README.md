# R-CNN for Cell Detection and Labeling Tool

The standard way to quantify biological imaging results is to count manually the number of each cell type in each image.
This manual counting is labor-intensive and often inconsistent. To do the counts automatically and deterministically, 
I used a deep learning model (R-CNN). The main goal of this R-CNN is to be able to replace any lab worker to 
create much faster and more reliable results. To create the data I also developed a labeling tool.

![Instance Segmentation Sample](assets/General_cell_detetion.JPG)

The repository includes:
* Labeling_Tool - 
    * Code for labeling tool.
* R_CNN - 
    * Training code on biological data.
    * Code for detect cells in new images.
    * Weights of the model trained on my dataset (biological images)
    * Pre-trained weights of MS COCO for the transfer learning.
    * Source code of Mask R-CNN built on FPN and ResNet101.
* User_Example -
    * Demonstration of my model on a new image


# Labeling Tool
In my study, I performed immunohistochemistry staining of biopsies taken from human stomachs. Therefore, the data 
consists of images while each image consists of three channels- blue, green, and red where each represents a different 
cell type. The main goal is to count the number of cell in each of the channels.

For example, in this image, the blue channel represents cells expressing CHGA/SYP, the red channel represents cells 
expressing TPH1/5HT and the green channel represents cells expressing HDC. The purple cells represent cells expressing
both CHGA/SYP and TPH1/5HT.
![Instance Segmentation Sample](assets/All_three_channels.JPG)

To create the data for the R-CNN, I created a tool called labeling tool. This tool allows the cells in the images to be 
easily labeled and counted because it provides most of the functions required for correct labeling, such as by marking the 
bounding box on the image, changing the brightness and the contrast of the image, and facilitates a convenient 
transition between all channels. In addition, the results are saved in such a way that they can later be used by the 
R-CNN.

The labeling tool:
![Instance Segmentation Sample](assets/Labeling_tool.JPG)
The tool displays two windows; one window presents the name of the image as well as the image itself, while the second 
window is the control window which allows the transition between the channels. The user can activate the channels of 
the image by left-clicking the control window. The blue background color represents a channel currently displayed in 
the image.

For example, while only the blue channel is selected in the control window, in the image window, the image only 
displays the blue channel, while in the control window, only the blue channel is displayed with a blue background.
In addition, two cells marked by the user by a bounding box.
![Instance Segmentation Sample](assets/Labeling_tool_blue.JPG)


# R-CNN for Cell Detection
All the results obtained from the labeling tool were transferred to the R-CNN model which generates bounding boxes for 
each instance of an object in the image. The images i used were in black-and-white and represent only a blue, red, or 
green channel. Each image was tagged according to the relevant channel.
Because the amount of data was not very large (750 images), I used transfer learning (using a pre-trained weights for 
MS COCO object detection dataset) to achieve faster and more accurate learning without overfitting but the 
learning can also be from scratch.

The images following displays marked cells in a given image. The image on the right is the network’s prediction. 
The image on the left contains the actual tagging done by a lab worker.
![Instance Segmentation Sample](assets/RCNN_vs_predicted_1.JPG)

![Instance Segmentation Sample](assets/RCNN_vs_predicted_2.JPG)


The image following displays marked cells in new images therefore there is no actual labeling. 
![Instance Segmentation Sample](assets/RCNN_on_new_images.JPG)

# User Example
This folder illustrates an example of running the model I trained. The folder contains a jupyter file and a folder 
contains one image. In order to run the example, the user should update the path of the folder where the images we 
want to examine are located and then run the program. The output for each image is its division into three channels 
and in each channel the number of cells detected.

If there are several images in the folder (more than one) then, in each image the cells will be displayed by bounding 
boxes and the number of cells identified in it will be written. Once the program finished plotting all the images,
the final number of cells in each of the channels will be printed.

## Requirements
Python 3.4, TensorFlow 1.15.3, Keras 2.2.4 and other common packages listed in `requirements.txt`.

## Acknowledgments
The R-CNN architecture in this project is based on [Mask R-CNN](https://github.com/matterport/Mask_RCNN).