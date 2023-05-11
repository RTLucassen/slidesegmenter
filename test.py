import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
from slidesegmenter import SlideSegmenter

directory = r'C:\Users\rlucasse\Documents\projects\tissue_segmentation\intermediate\data\images\test cases'

segmenter = SlideSegmenter()

for file in os.listdir(directory):
    path = os.path.join(directory, file)

    image = sitk.GetArrayFromImage(sitk.ReadImage(path))/255

    prediction = segmenter.segment(image)
    plt.imshow(prediction)
    plt.show()