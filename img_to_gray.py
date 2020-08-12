import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def image2gray(image):
    width, height = image.size[0], image.size[1]
    matrix_gray = np.zeros( (width, height) )

    for i in range(0, height):
        for j in range(0, width):
            red, green, blue = image.getpixel((i, j))
            pixel_gray = int(math.ceil((red * 0.298936) + (green * 0.587043) + (blue *0.114021)))
            pixel_gray /= math.pow(2, 8)
            matrix_gray[j][i] = pixel_gray

    return matrix_gray


im = Image.open("img/2.jpg").convert('RGB')
im_gray = image2gray(im)
plt.imshow(im_gray, cmap="gray")
plt.show()
