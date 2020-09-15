import cv2


def closing(im, size):
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    closing = cv2.morphologyEx(im, cv2.MORPH_CLOSE, struct)
    return closing
