from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2


def detect_glare(image: np.ndarray) -> np.ndarray:
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # threshold the image to reveal light regions in the
    # blurred image
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)

    # find the contours in the mask, then sort them from left to
    # right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if not cnts:
        return image
    else:
        cnts = contours.sort_contours(cnts)[0]
    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(cX), int(cY)), int(radius),
                   (0, 0, 255), 3)
        cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return image



def shadow_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = gray.mean() - 1.5 * gray.std()
    mask = np.zeros(gray.shape)
    mask[gray <= th] = 1
    return mask

def blue_detection(img):
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    th = blue.mean() + 2 * blue.std()
    mask = np.zeros(blue.shape)
    trsh = ((green >= th) + (red >= th)).astype(bool)
    th = (blue >= th).astype(bool)
    trsh = np.logical_xor(th, trsh)
    trsh = np.logical_and(th, trsh)
    mask[trsh] = 1
    return mask

def green_detection(img):
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    th = green.mean() + 2 * green.std()
    mask = np.zeros(green.shape)
    trsh = ((blue >= th) + (red >= th)).astype(bool)
    th = (green >= th).astype(bool)
    trsh = np.logical_xor(th, trsh)
    trsh = np.logical_and(th, trsh)
    mask[trsh] = 1
    return mask

def red_detection(img):
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    th = red.mean() + 2 * red.std()
    mask = np.zeros(red.shape)
    trsh = ((green >= th) + (blue >= th)).astype(bool)
    th = (red >= th).astype(bool)
    trsh = np.logical_xor(th, trsh)
    trsh = np.logical_and(th, trsh)
    mask[trsh] = 1
    return mask