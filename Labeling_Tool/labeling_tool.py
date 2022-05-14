# This tool is designed to label cells by bounding box and count them in each of the channels

# Imports
from Labeling_Tool import read_files
import cv2
import argparse
import numpy as np
import pickle
import os

# initialize the list of reference point
ref_point = []
images_dict = {}
alpha = 1.0
beta = 0
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
button_font_size = 1.5
white = (255, 255, 255)
black = (0)
blue = (100, 100, 100)
light_green = (200, 150, 0)
green = (0, 255, 0)
lineType = 2
b_button = [0, 30, 20, 100]
g_button = [0, 30, 120, 200]
r_button = [40, 70, 20, 100]
all_button = [40, 70, 120, 200]
control_image = np.zeros((80, 220, 3), np.uint8)
channel = "All channels"
image_name = ""


def draw_image(alpha, beta):
    global control_image, channel, curr_rect
    # create updated image
    image_with_rects = image.copy()
    for point in curr_rect:
        cv2.rectangle(image_with_rects, point[0], point[1], green, lineType)
    image_with_rects = cv2.convertScaleAbs(image_with_rects, alpha=alpha, beta=beta)
    cv2.putText(image_with_rects, channel, (10, 30), font, fontScale, white, lineType)
    cv2.putText(image_with_rects, str(len(curr_rect)), (10, 60), font, fontScale, white, lineType)
    cv2.imshow(image_name, image_with_rects)
    cv2.imshow('Control', control_image)


def draw_control():
    global b_flag, g_flag, r_flag, control_image
    # create button
    b_flag = g_flag = r_flag = True
    set_button_color_text(b_button, light_green, 'Blue', (35, 25))
    set_button_color_text(g_button, light_green, 'Green', (122, 25))
    set_button_color_text(r_button, light_green, 'Red', (35, 65))
    set_button_color_text(all_button, blue, 'All', (145, 65))
    # show 'control panel'
    cv2.imshow('Control', control_image)


def set_button_color_text(button, color_val, text, pos):
    global control_image
    control_image[button[0]:button[1], button[2]:button[3]] = color_val
    cv2.putText(control_image, text, pos, cv2.FONT_HERSHEY_PLAIN, button_font_size, black, lineType)


def shape_selection(event, x, y):
    """
    This method receives an event (mouse's click) and x,y coordinates (location) and accordingly updates the image
    and the list of all the rectangles
    """
    # grab references to the global variables
    global ref_point, curr_rect

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))
        # adds the rect to the rectangles list
        curr_rect.append(ref_point)
        # draw a rectangle around the region of interest
        draw_image(alpha, beta)

    elif event == cv2.EVENT_RBUTTONUP:
        if len(curr_rect):
            curr_rect.pop()
        draw_image(alpha, beta)


def button_switch(event, x, y):
    """
    This method receives an event (mouse's click) and x,y coordinates (location) and accordingly changes the color
    of the button and updates the current channel and image accordingly
    """
    global channel, curr_rect, image, control_image, b_flag, g_flag, r_flag
    # check if the click is within the dimensions of the button
    if event == cv2.EVENT_LBUTTONUP:

        if b_button[1] > y > b_button[0] < x < b_button[3]:
            if b_flag:
                b_flag = False
                set_button_color_text(b_button, 230, 'Blue', (35, 25))
            else:
                b_flag = True
                set_button_color_text(b_button, light_green, 'Blue', (35, 25))
        elif g_button[1] > y > g_button[0] < x < g_button[3]:
            if g_flag:
                g_flag = False
                set_button_color_text(g_button, 230, 'Green', (122, 25))
            else:
                g_flag = True
                set_button_color_text(g_button, light_green, 'Green', (122, 25))
        elif r_button[1] > y > r_button[0] < x < r_button[3]:
            if r_flag:
                r_flag = False
                set_button_color_text(r_button, 230, 'Red', (35, 65))
            else:
                r_flag = True
                set_button_color_text(r_button, light_green, 'Red', (35, 65))
        elif all_button[1] > y > all_button[0] < x < all_button[3]:
            draw_control()
            channel = "All channels"

        if b_flag and g_flag and r_flag:
            channel = "All channels"
        elif not b_flag and not g_flag and not r_flag:
            channel = "All channels"
        elif b_flag and not g_flag and not r_flag:
            channel = "Blue"
        elif not b_flag and g_flag and not r_flag:
            channel = "Green"
        elif not b_flag and not g_flag and r_flag:
            channel = "Red"
        elif b_flag and g_flag and not r_flag:
            channel = "Blue-Green"
        elif not b_flag and g_flag and r_flag:
            channel = "Red-Green"
        elif b_flag and not g_flag and r_flag:
            channel = "Blue-Red"

    curr_rect = rects_all_channels[channel]
    image = images_dict[channel]
    draw_image(alpha, beta)


def one_channel_image(orig_image, cnl):
    """
    This method receives an image and a specific channel and returns a black and white image
    only of the requested channel.
    """
    channel_image = orig_image.copy()
    channel_image[:, :, 0] = channel_image[:, :, cnl]
    channel_image[:, :, 1] = channel_image[:, :, cnl]
    channel_image[:, :, 2] = channel_image[:, :, cnl]
    return channel_image


def two_channels_image(orig_image, cnl):
    """
    This method receives an image and a specific channel and returns an image of the two other channels
    """
    channel_image = orig_image.copy()
    channel_image[:, :, cnl] = 0
    return channel_image


def contrast_vals(val):
    global alpha
    alpha = val * 0.5


def brightness_vals(val):
    global beta
    beta = val


def save_images(saving_place, image_nm, image_mode):
    """
    This method receives a name of the image, the mode of the image (whether in a folder or as an individual) and path
    and saves the image in this path
    """
    if image_mode:
        path_to_save = saving_place + "\\" + os.path.basename(os.path.normpath(image_nm))
    else:
        path_to_save = saving_place + "\\" + image_nm
    # cv2.imwrite(path_to_save, images_dict["All channels"])
    outfile = open(path_to_save[:-4] + "_results", 'wb')
    pickle.dump(rects_all_channels, outfile)
    outfile.close()


def run_program(img, name, saving_place, image_mode):
    global images_dict, curr_rect, image, image_name
    image_name = name
    # to display blue channel in black white
    blue_image = one_channel_image(img, 0)
    green_image = one_channel_image(img, 1)
    red_image = one_channel_image(img, 2)
    b_g_image = two_channels_image(img, 2)
    b_r_image = two_channels_image(img, 1)
    r_g_image = two_channels_image(img, 0)
    images_dict = {"All channels": img.copy(), "Blue": blue_image, "Green": green_image,
                   "Red": red_image, "Blue-Green": b_g_image, "Blue-Red": b_r_image, "Red-Green": r_g_image}

    # create an image window and attach a mousecallback and a trackbar
    cv2.namedWindow(image_name)
    cv2.createTrackbar("Contrast", image_name, 2, 20, contrast_vals)
    cv2.createTrackbar("Brightness", image_name, 4, 100, brightness_vals)
    cv2.setMouseCallback(image_name, shape_selection)
    # create a control window and attach a mousecallback and a trackbar
    cv2.namedWindow('Control')
    cv2.setMouseCallback('Control', button_switch)
    draw_control()
    exit_flag = False

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        key = cv2.waitKey(1) & 0xFF
        curr_rect = rects_all_channels[channel]
        image = images_dict[channel]
        draw_image(alpha, beta)
        if key == ord("q"):
            save_images(saving_place, image_name, image_mode)
            # print(rects_all_channels)
            break
        if key == ord("e"):
            save_images(saving_place, image_name, image_mode)
            # print(rects_all_channels)
            exit_flag = True
            break

    # close all open windows
    cv2.destroyAllWindows()
    return exit_flag


if __name__ == '__main__':
    # read the args - folder always
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=False, help="Path to the folder")
    ap.add_argument("-i", "--image", required=False, help="Path to the image")
    ap.add_argument("-s", "--save", required=True,
                    help="Path to the result")
    args = vars(ap.parse_args())

    # check if specific image was entered (if yes- run only the image, if no- go over all the folder)
    if args["image"] is None:
        images_list, images_name = read_files.open_folder(args["folder"])
        for i, img in enumerate(images_list):
            rects_all_channels = {"All channels": [], "Blue": [], "Green": [], "Red": [], "Blue-Green": [],
                                  "Blue-Red": [],
                                  "Red-Green": []}
            exit = run_program(img, images_name[i], args["save"], False)

            # after finish counting in the image, delete it from the original folder
            read_files.delete_image(args["folder"], images_name[i], img)
            if exit:
                break
    elif args["folder"] is None:
        rects_all_channels = {"All channels": [], "Blue": [], "Green": [], "Red": [], "Blue-Green": [], "Blue-Red": [],
                              "Red-Green": []}
        img = cv2.imread(args["image"])
        run_program(img, args["image"], args["save"], True)
    else:
        print("You can't write both folder and image name")
