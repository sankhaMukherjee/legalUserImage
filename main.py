import numpy             as np
import mediapipe         as mp
import matplotlib.pyplot as plt

from utils import (
    poseDetector, poseUtils
)

def main():

    fileName = 'data/Question 19 - 8AE44065-F2F5-4D85-9A96-F69471837F7A.jpeg'
    image    = poseUtils.getImage(fileName)

    pDetector      = poseDetector.PoseDetector()
    results        = pDetector.getResults( image )
    annotatedImage = poseUtils.generateAnnotatedImage( image, results )

    print(results.segmentation_mask)
    print(results.segmentation_mask.shape)


    plt.figure()
    plt.imshow(image)

    plt.figure()
    plt.imshow(annotatedImage)
    plt.savefig('results/temp.png')


    return

if __name__ == '__main__':
    main()
