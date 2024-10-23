import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def stereo(image_name, left_image, right_image, min_disparity = 0, max_disparities = 1, window_size = 3):

    # StereoSGBM parameters
    num_disparities = 16 * max_disparities  # Should be divisible by 16

    result_dir = 'result/Stereo'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Created directory: {result_dir}")

    # Create the stereo block matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # Compute disparity map
    disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0



    # Display the disparity map
    plt.figure(figsize=(10, 5))
    plt.imshow(disparity_map, 'gray')
    plt.title("Disparity Map")
    plt.colorbar()
    plt.savefig(os.path.join(result_dir, f'grayscale-Stereo-{image_name}-windows{window_size}-disparity{max_disparities}.png'))
    plt.show()



    disparity_map_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_map_normalized = np.uint8(disparity_map_normalized)

    # Apply a colormap
    color_disparity_map = cv2.applyColorMap(disparity_map_normalized, cv2.COLORMAP_JET)

    # Display using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(color_disparity_map, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display in Matplotlib
    plt.title("Disparity Map with Color Map (JET)")
    plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(result_dir, f'color-Stereo-{image_name}-windows{window_size}-disparity{max_disparities}.png'))
    plt.show()

    # Generate 3D point cloud from disparity map
    h, w = left_image.shape[:2]
    focal_length = 0.8 * w  # Adjust based on your camera
    baseline = 1.0  # Adjust based on your camera

    # Correct Q matrix
    Q = np.float32([[1, 0, 0, -w / 2.0],
                    [0, -1, 0, h / 2.0],
                    [0, 0, 0, focal_length],
                    [0, 0, -1 / baseline, 0]])

    # Reproject to 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

    # Mask to keep only valid points
    mask = disparity_map > 0
    points_3D = points_3D[mask]
    colors = left_image[mask]

    # Check the range of points_3D
    print("X range:", np.min(points_3D[:, 0]), np.max(points_3D[:, 0]))
    print("Y range:", np.min(points_3D[:, 1]), np.max(points_3D[:, 1]))
    print("Z range:", np.min(points_3D[:, 2]), np.max(points_3D[:, 2]))

    # Downsample for visualization (optional)
    step = 1
    x = points_3D[::step, 0]
    y = points_3D[::step, 1]
    z = points_3D[::step, 2]
    c = colors[::step] / 255.0

    # If not downsampling
    x = points_3D[:, 0]
    y = points_3D[:, 1]
    z = points_3D[:, 2]
    c = colors / 255.0

    # Normalize the data
    def safe_normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_max - arr_min == 0:
            return arr - arr_min
        else:
            return (arr - arr_min) / (arr_max - arr_min)

    x_norm = safe_normalize(x)
    y_norm = safe_normalize(y)
    z_norm = safe_normalize(z)

    fig = plt.figure(figsize=(18, 6))


    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(x_norm, y_norm, z_norm, c=c, s=0.1, alpha=0.6)
    ax1.set_title('Viewpoint 1')
    ax1.view_init(elev=20, azim=-60)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')


    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(x_norm, y_norm, z_norm, c=c, s=0.1, alpha=0.6)
    ax2.set_title('Viewpoint 2')
    ax2.view_init(elev=90, azim=0)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_zlim(0, 1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')


    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(x_norm, y_norm, z_norm, c=c, s=0.1, alpha=0.6)
    ax3.set_title('Viewpoint 3')
    ax3.view_init(elev=0, azim=0)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_zlim(0, 1)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'Steereo-{image_name}-3d_point_cloud.png'))
    plt.show()
