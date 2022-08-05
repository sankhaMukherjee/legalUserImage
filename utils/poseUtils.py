import cv2
import numpy as np

def getImage(fileName:str):

    image = cv2.imread( fileName )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    height, width, _ = image.shape
    if width > height:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image

def generateAnnotatedImage(image, results):

    BG_COLOR    = (192, 192, 192) # gray
    condition   = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image    = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotatedImage = image.copy()
    annotatedImage = np.where(condition, annotatedImage, bg_image)

    return annotatedImage
