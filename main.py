import cv2
from disparity import disparity
from Stereo import stereo

if __name__ == '__main__':
    # Load the left and right images
    image_name = "triclopsi2" # triclopsi2 corridor
    left_image_path = f"input/{image_name}l.jpg"
    right_image_path = f"input/{image_name}r.jpg"

    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # Run the function:
    for w in [9, 15, 21, 27]:
        for d in range(1):
            for method in ["sad", "ssd", "ncc"] :
                # If the color is True, the output will be color image, otherwise will be grayscale image.
                disparity(
                    method=method, 
                    window_size=w, 
                    max_disparity=d + 1, 
                    image_name=image_name, 
                    left_image=left_image, 
                    right_image=right_image,
                    color = True
                )

    stereo(image_name=image_name,left_image=left_image,right_image=right_image)