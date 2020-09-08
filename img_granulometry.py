import numpy as np
from scipy import ndimage


def granulometry_opening(data, size=None):
    size = np.ones((size,size)).astype(int)
    if size is None:
        size = np.ones((4,4)).astype(int)
    granulo_opening = ndimage.binary_opening(data, structure=size)
    return granulo_opening

def granulometry_closing(data, size=None):
    size = np.ones((size,size)).astype(int)
    if size is None:
        size = np.ones((4,4)).astype(int)
    granulo_closing = ndimage.binary_closing(data, structure=size)
    return granulo_closing



# mask = im > im.mean()
# granulo = granulometry_opening(mask, sizes=np.arange(2, 19, 4))
