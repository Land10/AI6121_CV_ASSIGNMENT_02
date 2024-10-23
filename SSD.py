import cv2
import numpy as np
import os
from tqdm import tqdm

def ssd(image_name, left_image, right_image, window_size = 5, max_disparity = 1):
    # Set parameter
    max_disparity = 20 * max_disparity
    height, width = left_image.shape
    half_window = window_size // 2

    # initialization
    disparity_map = np.zeros((height, width), dtype=np.uint8)

    # trivial image
    print(f"SSD with window size {window_size}, max disparity {max_disparity}")
    for y in tqdm(range(half_window, height - half_window), desc="Processing rows", unit="row"):
        for x in range(half_window, width - half_window):
            min_ssd = float('inf')
            best_disparity = 0
            for d in range(max_disparity):
                ssd = 0
                # make sure box not excess the range
                if x - d - half_window < 0:
                    break

                # calculation the SSD
                left_patch = left_image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
                right_patch = right_image[y - half_window:y + half_window + 1, x - half_window - d:x + half_window + 1 - d]
                ssd = np.sum(np.square(left_patch - right_patch))

                # update the minimal SSD
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_disparity = d

            # get the disparity
            disparity_map[y, x] = best_disparity * (255 // max_disparity)
            # disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
            # disparity_map = cv2.medianBlur(disparity_map, 5)  # 5x5 中值滤波
            color_disparity_map = cv2.applyColorMap(disparity_map, cv2.COLORMAP_JET)

    # Save the image
    output_dir = f'result/color-SSD-{image_name}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'disparity-windowsize{window_size}-maxdisparity-{max_disparity}.png')
    cv2.imwrite(output_path, color_disparity_map)

    return disparity_map




