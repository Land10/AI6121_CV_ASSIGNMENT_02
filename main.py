import cv2
import os
from SAD import sad
from SSD import ssd
from Stereo import stereo

if __name__ == '__main__':
    # Load the left and right images
    left_image_path = "input/triclopsi2l.jpg"
    right_image_path = "input/triclopsi2r.jpg"

    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    
    image_name = "triclopsi2"

    # Run the function:
    for w in [13, 21, 29]:
        for d in range(1):
            ssd(window_size=w, max_disparity=d + 1, image_name=image_name, left_image=left_image, right_image=right_image)

    # stereo(image_name=image_name,left_image=left_image,right_image=right_image)