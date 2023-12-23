# import the necessary packages
import numpy as np
import argparse
import cv2

def show_image(title, image, width = 300):
    # resize the image to have a constant width, just to
    # make displaying the images take up less screen real
    # estate
    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA).astype(np.uint8)
    
    # show the resized image
    plt.imshow(resized)
    plt.show()
    print(title)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# construct the argument parser and parse the arguments
# 
# load the images
source = cv2.imread("im1.jpeg")
# source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

target = cv2.imread("im2.jpeg")
# target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

# transfer the color distribution from the source image
# to the target image
transfer = color_transfer(source, target, clip=True, preserve_paper=False)

# check to see if the output image should be saved
# if args["output"] is not None:
#     cv2.imwrite(args["output"], transfer)

# show the images and wait for a key press
show_image("Source", source)
show_image("Target", target)
show_image("Transfer", transfer)