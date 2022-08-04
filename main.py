import cv2
import numpy             as np
import mediapipe         as mp
import matplotlib.pyplot as plt




def main():

    fileName = 'data/Question 19 - 8AE44065-F2F5-4D85-9A96-F69471837F7A.jpeg'
    image = cv2.imread( fileName )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    height, width, _ = image.shape
    if width > height:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    pose = mp.solutions.pose.Pose(
        static_image_mode        = True,
        model_complexity         = 2,
        enable_segmentation      = True,
        min_detection_confidence = 0.5,
    )

    results = pose.process( image )
    print(results.segmentation_mask)
    print(results.segmentation_mask.shape)

    BG_COLOR    = (192, 192, 192) # gray
    condition   = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image    = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = image.copy()
    annotated_image = np.where(condition, annotated_image, bg_image)

    plt.figure()
    plt.imshow(image)

    plt.figure()
    plt.imshow(annotated_image)
    plt.show()

    return

if __name__ == '__main__':
    main()
