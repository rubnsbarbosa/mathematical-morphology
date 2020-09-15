import cv2


def erosion(im, size):
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    img_erosion = cv2.erode(im, struct, iterations=1)
    return img_erosion