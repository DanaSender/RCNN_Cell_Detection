import cv2
import os

# Opening the new folder is for a situation where you open the folder of the photos, go over some image and then finish.
# So when we open the original folder again, we will not have to go through all the pictures we have already gone
# through. What the software does is create a folder and then move to it the files we have already gone through and
# delete them from the original folder.


def open_folder(folder):
    # in case the folder does not exists
    if not os.path.exists(folder+"\\images_already_finish"):
        os.mkdir(folder+"\\images_already_finish")
    images = []
    images_name = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img_name = filename
        if img is not None:
            images.append(img)
            images_name.append(img_name)
    return images, images_name


def delete_image(folder, image_name, img):
    if image_name in os.listdir(folder):
        os.remove(folder + "\\" + image_name)
        cv2.imwrite(folder+"\\images_already_finish\\"+image_name, img)
    else:
        print("The file does not exist")


