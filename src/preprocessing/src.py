import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def preprocess(cv_image: cv.typing.MatLike, debug=False) -> cv.typing.MatLike:
    """Preprocess an image for OCR.

    Args:
        cv_image (cv.typing.MatLike): The image to be preprocessed. MatLike is the type
        returned by cv.imread() and cv.VideoCapture.read().

    Returns:
        cv.typing.MatLike: Proprocessed image.

    Steps performed:
        1. Convert the image to grayscale. [Self.grayscale()]
        2. Apply a threshold to the image. [Self.threshold()]
        3. Remove small islands of noise. [Self.remove_islands()]
    """
    grayscaled = grayscale(cv_image, debug)
    
    if debug:
        cv.imshow("Grayscaled", grayscaled)

    thresholded = threshold(grayscaled, debug)
    
    if debug:
        cv.imshow("Thresholded", thresholded)

    removed_islands = remove_islands(thresholded, False)

    return removed_islands


def grayscale(cv_image: cv.typing.MatLike, debug=False) -> cv.typing.MatLike:
    return cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)


def threshold(cv_image: cv.typing.MatLike, debug=False) -> cv.typing.MatLike:
    # NOTE: According to https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html,
    # adaptive guassian thresholding is the most consistent and reliable method.
    # More research is required and tuning for the "C" and "blockSize" constants is required.
    thresholded = cv.adaptiveThreshold(
        cv_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 11
    )

    return thresholded


def remove_islands(cv_image: cv.typing.MatLike, debug=False) -> cv.typing.MatLike:

    # Find contours
    (numLabels, labels, stats, centroids) = cv.connectedComponentsWithStats(
        cv.bitwise_not(cv_image)
    )

    # Parts to be removed are 0.
    remove_mask = np.full(cv_image.shape, 255, dtype=np.uint8)

    # perform statistical analysis and decide on a removal size based on standard deviations
    # Calculate mean and standard deviation of areas
    areas = stats[:, cv.CC_STAT_AREA][1:]  # 1: to Skip the background label
    
    # A threshold under which to remove islands.
    # This will allow the statistics to do a better job
    # of adapting if the data is not flooded with small islands.
    hardcoded_threshold = 5

    # Filter out the super small islands first
    for i in range(1, numLabels):
        if stats[i, cv.CC_STAT_AREA] < hardcoded_threshold:
            remove_mask[labels == i] = 0
    
    areas = areas[areas >= hardcoded_threshold]

    mean_area = np.mean(areas.astype(np.float64))
    std_area = np.std(areas.astype(np.float64))
    
    if debug:
        cv.imshow("Removed islands hardcoded", remove_mask)
    
    if debug:
        print(f"Mean area: {mean_area}, Standard deviation: {std_area}")

    # Set a threshold for removal
    k = 1
    removal_threshold = mean_area - k * std_area
    if debug:
        print(f"Removal threshold: {removal_threshold}")
        
    if removal_threshold > hardcoded_threshold:
        for i in range(1, numLabels):
            # If the area is less than the threshold, set to 0.
            if stats[i, cv.CC_STAT_AREA] < removal_threshold:
                remove_mask[labels == i] = 0
        
        if debug:
            print(f"Removed islands with statistical analysis")
            cv.imshow("Removed islands statistically", remove_mask)

    # If the pixels are the same, set to 255.
    img_clone = cv_image.copy()

    img_clone[remove_mask == 0] = 255

    return img_clone
