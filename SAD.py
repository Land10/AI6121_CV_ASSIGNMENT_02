import cv2
import numpy as np
import os
from tqdm import tqdm

def sad(image_name, left_image, right_image, window_size = 5, max_disparity = 1):
    # Set parameter
    max_disparity = 16 * max_disparity
    height, width = left_image.shape
    half_window = window_size // 2

    # initialization
    disparity_map = np.zeros((height, width), dtype=np.uint8)

    # trivial image
    print(f"SAD with window size {window_size}, max disparity {max_disparity}")
    for y in tqdm(range(half_window, height - half_window), desc="Processing rows", unit="row"):
        for x in range(half_window, width - half_window):
            min_sad = float('inf')
            best_disparity = 0
            for d in range(max_disparity):
                sad = 0
                # make sure box not excess the range
                if x - d - half_window < 0:
                    break

                # calculation the SAD
                left_patch = left_image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
                right_patch = right_image[y - half_window:y + half_window + 1, x - half_window - d:x + half_window + 1 - d]
                sad = np.sum(np.abs(left_patch - right_patch))
                # for v in range(-half_window, half_window + 1):
                #     for u in range(-half_window, half_window + 1):
                #         left_pixel = int(left_image[y + v, x + u])
                #         right_pixel = int(right_image[y + v, x + u - d])
                #         diff = left_pixel - right_pixel
                #         sad += abs(diff)

                # update the minimal SAD
                if sad < min_sad:
                    min_sad = sad
                    best_disparity = d

            # get the disparity
            disparity_map[y, x] = best_disparity * (255 // max_disparity)

    # Save the image
    output_dir = f'result/SAD-{image_name}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'disparity-windowsize{window_size}-maxdisparity-{max_disparity}.png')
    cv2.imwrite(output_path, disparity_map)

    return disparity_map




