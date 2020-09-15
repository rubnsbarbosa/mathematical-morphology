import cv2


def opening(im, size):
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    opening = cv2.morphologyEx(im , cv2.MORPH_OPEN, struct)
    return opening