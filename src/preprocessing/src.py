import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class PreprocessPreset(Enum):
    """Preprocessing presets."""

    ADAPTIVE = 1
    NOISY = 2
    CLEAN = 3
    TOPHAT_MORPH = 4


class Preprocessor:
    def __init__(self, preset: PreprocessPreset, approx_char_size: int = 20):
        if approx_char_size < 10:
            raise ValueError("Approximate character size must be greater than 9.")

        self.preset = preset
        self.approx_char_size = approx_char_size

    def preprocess(self, cv_image: cv.typing.MatLike, debug=False) -> cv.typing.MatLike:
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
        grayscaled = self.grayscale(cv_image, debug)

        if debug:
            cv.imshow("Grayscaled", grayscaled)

        thresholded = self.threshold(grayscaled, debug)

        if debug:
            cv.imshow("Thresholded", thresholded)

        removed_islands = self.remove_islands(thresholded, debug)

        return removed_islands

    def top_hat_morph(
        self, cv_image: cv.typing.MatLike, debug=False
    ) -> cv.typing.MatLike:
        morphed = cv.morphologyEx(
            cv_image, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        )

        return morphed

    def grayscale(self, cv_image: cv.typing.MatLike, debug=False) -> cv.typing.MatLike:
        return cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)

    def threshold(self, cv_image: cv.typing.MatLike, debug=False) -> cv.typing.MatLike:
        # NOTE: According to https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html,
        # adaptive guassian thresholding is the most consistent and reliable method.
        # More research is required and tuning for the "C" and "blockSize" constants is required.

        # Plot a histogram of pixel counts binning by 8
        # plt.hist(cv_image.ravel(), 256, [0, 180])
        # plt.show()
        
        # Find the first peak, this will be the most common dark color
        # This will be used as the threshold for the non-adaptive thresholding
        hist, bins = np.histogram(cv_image.ravel(), 256, [0, 156])
        
        # Compute the bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute the mean of the histogram
        mean = np.sum(hist * bin_centers) / np.sum(hist)

        # Compute the variance
        variance = np.sum(hist * (bin_centers - mean) ** 2) / np.sum(hist)

        # Compute the standard deviation
        std_dev = np.sqrt(variance)
        
        # Find the first peak
        first_peak = np.argmax(hist)
        
        if debug:
            print(f"First peak: {first_peak}, Standard deviation: {std_dev}")
        
        # Set the threshold to the first peak
        threshold = first_peak + std_dev
        
        # Remove pixels above the threshold
        cv_image = cv.threshold(cv_image, threshold, 255, cv.THRESH_BINARY)[1]

        cv.imshow("Thresholded Non-adaptive", cv_image)

        blockSize = self.approx_char_size

        if blockSize % 2 == 0:
            blockSize -= 1

        thresholded = cv.adaptiveThreshold(
            cv_image,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            blockSize,
            16,
        )

        return thresholded

    def remove_islands(
        self, cv_image: cv.typing.MatLike, debug=False
    ) -> cv.typing.MatLike:

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
        area_threshold = 2 + ((self.approx_char_size // 8) ** 2)

        if area_threshold <= 4:
            area_threshold = 5

        if debug:
            print(f"Area threshold: {area_threshold}")

        # Filter out the super small islands first
        for i in range(1, numLabels):
            if stats[i, cv.CC_STAT_AREA] < area_threshold:
                remove_mask[labels == i] = 0

        areas = areas[areas >= area_threshold]

        mean_area = np.mean(areas.astype(np.float64))
        std_area = np.std(areas.astype(np.float64))

        if debug:
            print(f"Mean area: {mean_area}, Standard deviation: {std_area}")

        # Width and height removal
        max_1 = self.approx_char_size // 6
        max_2 = self.approx_char_size * 2

        for i in range(1, numLabels):
            if (
                # Remove long horizontal lines
                stats[i, cv.CC_STAT_HEIGHT] <= max_1
                and stats[i, cv.CC_STAT_WIDTH] <= max_2
                # Don't remove dots as they are important
                and stats[i, cv.CC_STAT_WIDTH] > max_1
            ):
                remove_mask[labels == i] = 0

        if debug:
            cv.imshow("Remove Mask", remove_mask)

        # If the pixels are the same, set to 255.
        img_clone = cv_image.copy()

        img_clone[remove_mask == 0] = 255

        # Straight line removal if way outside of the standard deviation
        # This removes any scratches or lines that are not part of the text.
        # This should be fine for burmese text as it does not contain straight lines.

        return img_clone
