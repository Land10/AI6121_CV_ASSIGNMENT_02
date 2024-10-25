import cv2
import numpy as np
import os


def Sobel_filter(image):
    myksize = 3
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=myksize)  # horizontal direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=myksize)  # vertical direction
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    Sobel_image = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    return Sobel_image


def post_processing(left_image_name, right_image_name, HE = True):
    left_image = cv2.imread(f'input/{left_image_name}')
    right_image = cv2.imread(f'input/{right_image_name}')

    # set the image to grayscale
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # HE 
    if HE:
        left_image = cv2.equalizeHist(left_image)
        right_image = cv2.equalizeHist(right_image)
    
    # Sobel filter
    left_image = Sobel_filter(left_image)
    right_image = Sobel_filter(right_image)

    # save the processed images
    if HE:
        cv2.imwrite(f'input/sobel_image/HE_{left_image_name}.png', left_image)
        cv2.imwrite(f'input/sobel_image/HE_{right_image_name}.png', right_image)
    else :
        cv2.imwrite(f'input/sobel_image/{left_image_name}.png', left_image)
        cv2.imwrite(f'input/sobel_image/{right_image_name}.png', right_image)

if __name__ == '__main__':
    # choose the image name
    left_image_name = 'triclopsi2l.jpg'
    right_image_name = 'triclopsi2r.jpg'
    # create the output directory if it doesn't exist
    output_dir = 'input/sobel_image/'
    os.makedirs(output_dir, exist_ok=True)
    # process the images
    post_processing(left_image_name, right_image_name, HE = True)
    post_processing(left_image_name, right_image_name, HE = False)
