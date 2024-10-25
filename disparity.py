import cv2
import numpy as np
import os
from tqdm import tqdm

def sad(left_patch, right_patch) :
    return np.sum(np.abs(left_patch - right_patch))

def ssd(left_patch, right_patch) :
    return np.sum(np.square(left_patch - right_patch))

def ncc(left_patch, right_patch) :
    left_mean = np.mean(left_patch)
    right_mean = np.mean(right_patch)
    
    numerator = np.sum((left_patch - left_mean) * (right_patch - right_mean))
    denominator = np.sqrt(np.sum(np.square(left_patch - left_mean)) * np.sum(np.square(right_patch - right_mean)))
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def disparity(method, image_name, left_image, right_image, window_size = 5, max_disparity = 1, color = False):
    # Set parameter
    max_disparity = 16 * max_disparity
    height, width = left_image.shape
    half_window = window_size // 2

    # initialization
    disparity_map = np.zeros((height, width), dtype=np.uint8)

    # trivial image
    print(f"{method} with window size {window_size}, max disparity {max_disparity}")
    for y in tqdm(range(half_window, height - half_window), desc="Processing rows", unit="row"):
        for x in range(half_window, width - half_window):
            min_value = float('inf')
            max_value = -float('inf')
            best_disparity = 0
            for d in range(max_disparity):
                # make sure box not excess the range
                if x - d - half_window < 0:
                    break

                # calculate and update the value
                left_patch = left_image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
                right_patch = right_image[y - half_window:y + half_window + 1, x - half_window - d:x + half_window + 1 - d]
                
                if method == "sad" :
                    sad_value = sad(left_patch, right_patch)
                    if sad_value < min_value:
                        min_value = sad_value
                        best_disparity = d
                elif method == "ssd" :
                    ssd_value = ssd(left_patch, right_patch)
                    if ssd_value < min_value:
                        min_value = ssd_value
                        best_disparity = d
                elif method == "ncc" :
                    ncc_value = ncc(left_patch, right_patch)
                    if ncc_value > max_value:
                        max_value = ncc_value
                        best_disparity = d
                else :
                    print("No such method!")
                    exit()

            # get the disparity
            disparity_map[y, x] = best_disparity * (255 // max_disparity)
            # set the color
            if color:
                disparity_map = cv2.applyColorMap(disparity_map, cv2.COLORMAP_JET)


    # Save the image
    output_dir = f'result/{method}-{image_name}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'disparity-windowsize{window_size}-maxdisparity-{max_disparity}.png')
    cv2.imwrite(output_path, disparity_map)
    
    # Denoise
    os.makedirs(f'result/{method}-{image_name}-denoised', exist_ok=True)
    output_dir = f'result/{method}-{image_name}-denoised/disparity-windowsize{window_size}-maxdisparity-{max_disparity}-'
    filtered_map = cv2.medianBlur(disparity_map, 5)
    cv2.imwrite(output_dir + "medianblur.png", filtered_map)
    filtered_map = cv2.GaussianBlur(disparity_map, (3, 3), 0)
    cv2.imwrite(output_dir + "gaussianblur.png", filtered_map)
    filtered_map = cv2.bilateralFilter(disparity_map, d=9, sigmaColor=25, sigmaSpace=25)
    cv2.imwrite(output_dir + "bilateralfilter.png", filtered_map)
    opened_map = cv2.morphologyEx(disparity_map, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cv2.imwrite(output_dir + "morphology.png", opened_map)

