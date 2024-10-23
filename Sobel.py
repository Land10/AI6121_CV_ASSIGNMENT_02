import cv2
import numpy as np
import os

def sobel(image_name, image): 
    # calculate the gradients using Sobel operator
    my_ksize = 3
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=my_ksize)  
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=my_ksize)  

    # gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

    # save the image
    output_dir = f'input/shadow_image'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{image_name}.png')
    cv2.imwrite(output_path, gradient_magnitude)

if __name__ == '__main__':
    # read the image
    left_image_path = "input/corridorl.jpg"
    right_image_path = "input/corridorr.jpg"
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE) 
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE) 

    left_image_name = 'processed_shadow_image_left'
    right_image_name = 'processed_shadow_image_right'
    
    sobel(left_image_name, left_image)
    sobel(right_image_name, right_image)