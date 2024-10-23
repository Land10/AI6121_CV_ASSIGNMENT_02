import cv2
import numpy as np

left_image_name = 'triclopsi2l.jpg'
right_image_name = 'triclopsi2r.jpg'
left_image = cv2.imread(f'input/{left_image_name}')
right_image = cv2.imread(f'input/{right_image_name}')

left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

left_image_eq = cv2.equalizeHist(left_image_gray)
right_image_eq = cv2.equalizeHist(right_image_gray)

# cv2.imshow('Enhanced Image with Laplacian Edges', left_image_eq)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

myksize = 3
sobel_x = cv2.Sobel(left_image_eq, cv2.CV_64F, 1, 0, ksize=myksize)  # horizontal direction
sobel_y = cv2.Sobel(left_image_eq, cv2.CV_64F, 0, 1, ksize=myksize)  # vertical direction

gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))


sobel_x = cv2.Sobel(right_image_eq, cv2.CV_64F, 1, 0, ksize=myksize)  # horizontal direction
sobel_y = cv2.Sobel(right_image_eq, cv2.CV_64F, 0, 1, ksize=myksize)  # vertical direction
gradient_magnitude2 = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
gradient_magnitude2 = np.uint8(255 * gradient_magnitude2 / np.max(gradient_magnitude2))


cv2.imwrite('input/sobel_image/HE_triclopsi2l.png', gradient_magnitude)
cv2.imwrite('input/sobel_image/HE_triclopsi2r.png',gradient_magnitude2)

sobel_x = cv2.Sobel(left_image_gray, cv2.CV_64F, 1, 0, ksize=myksize) 
sobel_y = cv2.Sobel(left_image_gray, cv2.CV_64F, 0, 1, ksize=myksize) 
gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
cv2.imwrite('input/sobel_image/triclopsi2l.png', gradient_magnitude)

sobel_x = cv2.Sobel(right_image_gray, cv2.CV_64F, 1, 0, ksize=myksize) 
sobel_y = cv2.Sobel(right_image_gray, cv2.CV_64F, 0, 1, ksize=myksize) 
gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
cv2.imwrite('input/sobel_image/triclopsi2r.png', gradient_magnitude)