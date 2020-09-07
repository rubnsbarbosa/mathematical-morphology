import cv2


def dilatation(im, size):
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    img_dilation = cv2.dilate(im, struct, iterations=1)
    return img_dilation
