import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from img_to_gray import image2gray
from img_dilatation import dilatation
from img_segmentation import segmentation_Kmeans


def convert_matrix2array(im_matrix):
    array = im_matrix.view(np.ndarray)
    array.shape = -1
    return array

## Dataframe with images, each column is a pixel and each row is an image
X_mal = pd.DataFrame()

for image in glob.glob("img/*.jpg"):
    try:
        # open image
        im = Image.open(image).convert('RGB')
        # image to gray scale
        im_gray = image2gray(im)
        # dilatation in image gray with structuring element 5x5
        im_dilate = dilatation(im_gray, 5)
        # applying segmentation
        im_seg = segmentation_Kmeans(im_dilate, num_centroids)
        # convert (image segmentated matrix) to array
        im_array = convert_matrix2array(im_seg)
        # creating dataset of images malignant
        X_malignant = pd.DataFrame(im_array).T
        # append in dataframe
        X_mal = X_mal.append(X_malignant, ignore_index=True)
    

    except:
        print("error")

### Save dataframe of malignant images in CSV file
X_mal.to_csv("malignant.csv", index=False)
### Read CSV
X_malig = pd.read_csv("malignant.csv")
X_malig.head()
