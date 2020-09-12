import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from img_to_gray import image2gray
from img_dilatation import dilatation
from img_segmentation import segmentation_Kmeans
from img_granulometry import granulometry_opening, granulometry_closing

directory = "./imgs"
# print(len(directory))
# im_list = [im for im in os.listdir(directory) if im.endswith('.jpg')]
# print(len(im_list))

num_centroids = 2
results = "./results"

for image in glob.glob("img/*.jpg"):
    try:
        file_name, ext = os.path.splitext(image)
        
        # open image
        img = Image.open(image).convert('RGB')
        img_gray = image2gray(img)
        # segmentation
        img_seg = segmentation_Kmeans(img_gray, num_centroids)
        # dilatation in image gray
        img_dilate = dilatation(img_gray, 5)
        # now, applying segmentation in image dilated
        img_seg_dilate = segmentation_Kmeans(img_dilate, num_centroids)
        
        # granulometry 
        #img_open = granulometry_opening(img_seg, size=3)
        #plt.contour(opened, colors='b', linewidths=2)
        #plt.show()

        # plots
        plt.figure(figsize=(12, 3.5))
        plt.title(file_name)
        plt.subplot(141)
        plt.imshow(img)
        plt.title('Original image')
        plt.axis('off')

        plt.subplot(142)
        plt.imshow(img_gray, cmap="gray")
        plt.title('Image gray')
        plt.axis('off')

        plt.subplot(143)
        plt.imshow(img_seg, cmap=plt.cm.gray)
        plt.title('Segmentation')
        plt.axis('off')

        plt.subplot(144)
        plt.imshow(img_seg_dilate, cmap=plt.cm.gray)
        plt.title('Segmentation & Dilatation')
        plt.axis('off')

        plt.subplots_adjust(wspace=.05, left=.01, bottom=.01, right=.99, top=.99)
        plt.savefig(results+file_name)
        #plt.show()
        
    
    except:
        print("error")
